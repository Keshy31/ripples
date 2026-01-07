#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

// Forward declaration of CUDA kernel
extern "C" __global__ void fused_update_kernel(float* prev_u, float* u, float* next_u, float c, float dt, float dx, float damping, float freq, float amp, float t, int source_x, int source_y, int size);

// Simulation parameters (global for key callback access)
struct SimParams {
    float freq = 10.0f;
    float amp = 5.0f;
    float c = 0.5f;
    // Defaults for reset
    static constexpr float DEFAULT_FREQ = 10.0f;
    static constexpr float DEFAULT_AMP = 5.0f;
    static constexpr float DEFAULT_C = 0.5f;
} g_params;

// Flag to clear simulation (checked in main loop)
bool g_clearSim = false;

// Camera state (spherical coordinates)
struct Camera {
    float yaw = 0.0f;           // Horizontal angle (radians)
    float pitch = 1.5f;         // Vertical angle (radians), ~86 degrees = almost top-down
    float distance = 5.0f;      // Distance from origin
    float lastX = 0.0f;
    float lastY = 0.0f;
    bool dragging = false;

    glm::vec3 getPosition() const {
        float x = distance * cos(pitch) * sin(yaw);
        float y = distance * sin(pitch);
        float z = distance * cos(pitch) * cos(yaw);
        return glm::vec3(x, y, z);
    }
} g_camera;

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    // Don't capture mouse if ImGui wants it
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return;

    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            g_camera.dragging = true;
            double xpos, ypos;
            glfwGetCursorPos(window, &xpos, &ypos);
            g_camera.lastX = static_cast<float>(xpos);
            g_camera.lastY = static_cast<float>(ypos);
        } else if (action == GLFW_RELEASE) {
            g_camera.dragging = false;
        }
    }
}

void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos) {
    if (!g_camera.dragging) return;

    float dx = static_cast<float>(xpos) - g_camera.lastX;
    float dy = static_cast<float>(ypos) - g_camera.lastY;
    g_camera.lastX = static_cast<float>(xpos);
    g_camera.lastY = static_cast<float>(ypos);

    const float sensitivity = 0.005f;
    g_camera.yaw += dx * sensitivity;
    g_camera.pitch += dy * sensitivity;

    // Clamp pitch to avoid flipping
    g_camera.pitch = std::max(0.1f, std::min(1.56f, g_camera.pitch)); // ~5 to ~89 degrees
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    // Don't capture scroll if ImGui wants it
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return;

    g_camera.distance -= static_cast<float>(yoffset) * 0.5f;
    g_camera.distance = std::max(1.0f, std::min(20.0f, g_camera.distance));
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        switch (key) {
            case GLFW_KEY_UP:
                g_params.freq += 1.0f;
                std::cout << "Frequency: " << g_params.freq << " Hz" << std::endl;
                break;
            case GLFW_KEY_DOWN:
                g_params.freq = std::max(1.0f, g_params.freq - 1.0f);
                std::cout << "Frequency: " << g_params.freq << " Hz" << std::endl;
                break;
            case GLFW_KEY_RIGHT:
                g_params.amp += 0.5f;
                std::cout << "Amplitude: " << g_params.amp << std::endl;
                break;
            case GLFW_KEY_LEFT:
                g_params.amp = std::max(0.5f, g_params.amp - 0.5f);
                std::cout << "Amplitude: " << g_params.amp << std::endl;
                break;
            case GLFW_KEY_W:
                g_params.c += 0.1f;
                std::cout << "Wave speed: " << g_params.c << std::endl;
                break;
            case GLFW_KEY_S:
                g_params.c = std::max(0.1f, g_params.c - 0.1f);
                std::cout << "Wave speed: " << g_params.c << std::endl;
                break;
            case GLFW_KEY_R:
                g_params.freq = SimParams::DEFAULT_FREQ;
                g_params.amp = SimParams::DEFAULT_AMP;
                g_params.c = SimParams::DEFAULT_C;
                std::cout << "Reset to defaults: freq=" << g_params.freq
                          << " amp=" << g_params.amp << " c=" << g_params.c << std::endl;
                break;
            case GLFW_KEY_C:
                g_clearSim = true;
                std::cout << "Clearing simulation..." << std::endl;
                break;
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, GLFW_TRUE);
                break;
        }
    }
}

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Set OpenGL version
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create a windowed mode window and its OpenGL context
    const int windowWidth = 1920;
    const int windowHeight = 1080;
    GLFWwindow* window = glfwCreateWindow(windowWidth, windowHeight, "Ripples", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    // Register input callbacks
    glfwSetKeyCallback(window, key_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // Initialize GLAD
    if (!gladLoadGL(glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 450");

    // CUDA setup
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    std::cout << "Using GPU: " << deviceProp.name << std::endl;
    
    // Simulation constants
    const size_t grid_size = 2048;
    const float dx = 1.0f / static_cast<float>(grid_size - 1);
    const float damping = 0.01f;
    float t = 0.0f;

    // Print controls
    std::cout << "\nControls:" << std::endl;
    std::cout << "  Up/Down arrows: Frequency +/- 1 Hz" << std::endl;
    std::cout << "  Left/Right arrows: Amplitude +/- 0.5" << std::endl;
    std::cout << "  W/S: Wave speed +/- 0.1" << std::endl;
    std::cout << "  R: Reset to defaults" << std::endl;
    std::cout << "  C: Clear simulation" << std::endl;
    std::cout << "  ESC: Quit\n" << std::endl;
    
    // Device arrays
    float* device_prev_u = nullptr;
    float* device_u = nullptr;
    float* device_next_u = nullptr;
    cudaMalloc(&device_prev_u, grid_size * grid_size * sizeof(float));
    cudaMalloc(&device_u, grid_size * grid_size * sizeof(float));
    cudaMalloc(&device_next_u, grid_size * grid_size * sizeof(float));
    cudaMemset(device_prev_u, 0, grid_size * grid_size * sizeof(float));
    cudaMemset(device_u, 0, grid_size * grid_size * sizeof(float));
    cudaMemset(device_next_u, 0, grid_size * grid_size * sizeof(float));
    
    // OpenGL texture for displacement
    GLuint displacementTex;
    glGenTextures(1, &displacementTex);
    glBindTexture(GL_TEXTURE_2D, displacementTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, grid_size, grid_size, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    // CUDA-GL interop
    cudaGraphicsResource_t cudaResource;
    cudaGraphicsGLRegisterImage(&cudaResource, displacementTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
    
    // Shader program
    auto loadShader = [](const char* path, GLenum type) -> GLuint {
        std::ifstream file(path);
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string source = buffer.str();
        const char* src = source.c_str();
        GLuint shader = glCreateShader(type);
        glShaderSource(shader, 1, &src, nullptr);
        glCompileShader(shader);
        int success;
        char infoLog[512];
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 512, NULL, infoLog);
            std::cerr << "Shader compile error: " << infoLog << std::endl;
        }
        return shader;
    };
    
    // Helper to create shader program
    auto createProgram = [&loadShader](const char* vertPath, const char* fragPath) -> GLuint {
        GLuint vert = loadShader(vertPath, GL_VERTEX_SHADER);
        GLuint frag = loadShader(fragPath, GL_FRAGMENT_SHADER);
        GLuint prog = glCreateProgram();
        glAttachShader(prog, vert);
        glAttachShader(prog, frag);
        glLinkProgram(prog);
        int success;
        char infoLog[512];
        glGetProgramiv(prog, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(prog, 512, NULL, infoLog);
            std::cerr << "Program link error (" << fragPath << "): " << infoLog << std::endl;
        }
        return prog;
    };

    // Main scene shader
    GLuint program = createProgram("src/shaders/vertex.glsl", "src/shaders/fragment.glsl");

    // Post-processing shaders
    GLuint brightExtractProg = createProgram("src/shaders/fullscreen.vert", "src/shaders/bright_extract.frag");
    GLuint blurProg = createProgram("src/shaders/fullscreen.vert", "src/shaders/blur.frag");
    GLuint ssaoProg = createProgram("src/shaders/fullscreen.vert", "src/shaders/ssao.frag");
    GLuint compositeProg = createProgram("src/shaders/fullscreen.vert", "src/shaders/composite.frag");

    // Post-processing settings (global for ImGui access)
    struct PostProcessSettings {
        bool enableBloom = true;
        bool enableSSAO = true;
        bool enableFXAA = true;
        float bloomThreshold = 0.8f;
        float bloomIntensity = 0.3f;
        float ssaoRadius = 0.5f;
    } g_postProcess;

    // G-Buffer FBO (color + normals + depth)
    GLuint gBufferFBO, gColorTex, gNormalTex, gDepthTex;
    glGenFramebuffers(1, &gBufferFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, gBufferFBO);

    glGenTextures(1, &gColorTex);
    glBindTexture(GL_TEXTURE_2D, gColorTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gColorTex, 0);

    glGenTextures(1, &gNormalTex);
    glBindTexture(GL_TEXTURE_2D, gNormalTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, gNormalTex, 0);

    glGenTextures(1, &gDepthTex);
    glBindTexture(GL_TEXTURE_2D, gDepthTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, windowWidth, windowHeight, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, gDepthTex, 0);

    GLenum gBufferAttachments[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
    glDrawBuffers(2, gBufferAttachments);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cerr << "G-Buffer FBO incomplete!" << std::endl;

    // Bloom FBOs (ping-pong for blur)
    GLuint bloomFBO[2], bloomTex[2];
    glGenFramebuffers(2, bloomFBO);
    glGenTextures(2, bloomTex);
    for (int i = 0; i < 2; i++) {
        glBindFramebuffer(GL_FRAMEBUFFER, bloomFBO[i]);
        glBindTexture(GL_TEXTURE_2D, bloomTex[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, windowWidth / 2, windowHeight / 2, 0, GL_RGBA, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, bloomTex[i], 0);
    }

    // SSAO FBO
    GLuint ssaoFBO, ssaoTex;
    glGenFramebuffers(1, &ssaoFBO);
    glGenTextures(1, &ssaoTex);
    glBindFramebuffer(GL_FRAMEBUFFER, ssaoFBO);
    glBindTexture(GL_TEXTURE_2D, ssaoTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, windowWidth, windowHeight, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ssaoTex, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Empty VAO for fullscreen triangle
    GLuint fullscreenVAO;
    glGenVertexArrays(1, &fullscreenVAO);
    
    // 3D plane mesh
    std::vector<float> vertices;
    const int mesh_res = 256;
    for (int i = 0; i <= mesh_res; ++i) {
        for (int j = 0; j <= mesh_res; ++j) {
            float x = (float)j / mesh_res * 2.0f - 1.0f;
            float y = (float)i / mesh_res * 2.0f - 1.0f;
            vertices.insert(vertices.end(), {x, 0.0f, y});
        }
    }
    std::vector<unsigned int> indices;
    for (int i = 0; i < mesh_res; ++i) {
        for (int j = 0; j < mesh_res; ++j) {
            unsigned int idx = i * (mesh_res + 1) + j;
            indices.insert(indices.end(), {idx, idx + 1, idx + mesh_res + 1,
                                           idx + 1, idx + mesh_res + 2, idx + mesh_res + 1});
        }
    }
    
    GLuint VAO, VBO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Matrices (view computed per frame from camera state)
    glm::mat4 projection = glm::perspective(glm::radians(60.0f), (float)windowWidth / (float)windowHeight, 0.1f, 100.0f);
    glm::mat4 model = glm::scale(glm::mat4(1.0f), glm::vec3(4.0f, 1.0f, 4.0f));  // Scale mesh 4x to fill view
    glm::vec3 lightPos(2.0f, 5.0f, 2.0f);  // Offset light for better shading
    
    // Set the clear color
    glClearColor(0.02f, 0.02f, 0.05f, 1.0f);  // Dark background

    // After glad init
    glEnable(GL_DEPTH_TEST);

    // Loop until the user closes the window
    while (!glfwWindowShouldClose(window)) {
        // Simulate step (dt computed from current wave speed for CFL stability)
        float dt = dx / (g_params.c * 1.5f);
        dim3 block_dim(16, 16, 1);
        dim3 grid_dim((grid_size + 15) / 16, (grid_size + 15) / 16, 1);
        fused_update_kernel<<<grid_dim, block_dim>>>(device_prev_u, device_u, device_next_u, g_params.c, dt, dx, damping, g_params.freq, g_params.amp, t, grid_size / 2, grid_size / 2, grid_size);
        cudaDeviceSynchronize();
        t += dt;
    
        // Swap pointers
        float* temp = device_prev_u;
        device_prev_u = device_u;
        device_u = device_next_u;
        device_next_u = temp;

        // Clear simulation if requested
        if (g_clearSim) {
            cudaMemset(device_prev_u, 0, grid_size * grid_size * sizeof(float));
            cudaMemset(device_u, 0, grid_size * grid_size * sizeof(float));
            cudaMemset(device_next_u, 0, grid_size * grid_size * sizeof(float));
            t = 0.0f;
            g_clearSim = false;
        }

        // Update texture with CUDA
        cudaGraphicsMapResources(1, &cudaResource);
        cudaArray_t array;
        cudaGraphicsSubResourceGetMappedArray(&array, cudaResource, 0, 0);
        cudaMemcpy2DToArray(array, 0, 0, device_u, grid_size * sizeof(float), grid_size * sizeof(float), grid_size, cudaMemcpyDeviceToDevice);
        cudaGraphicsUnmapResources(1, &cudaResource);
    
        // Poll for and process events
        glfwPollEvents();

        // Compute view matrix from camera state
        glm::vec3 camPos = g_camera.getPosition();
        glm::mat4 view = glm::lookAt(camPos, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

        // ============ PASS 1: Render scene to G-Buffer ============
        glBindFramebuffer(GL_FRAMEBUFFER, gBufferFBO);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);

        glUseProgram(program);
        glUniformMatrix4fv(glGetUniformLocation(program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(glGetUniformLocation(program, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(program, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniform3fv(glGetUniformLocation(program, "lightPos"), 1, glm::value_ptr(lightPos));
        glUniform3fv(glGetUniformLocation(program, "camPos"), 1, glm::value_ptr(camPos));
        glUniform3fv(glGetUniformLocation(program, "lightColor"), 1, glm::value_ptr(glm::vec3(1.0f)));
        glBindVertexArray(VAO);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, displacementTex);
        glUniform1i(glGetUniformLocation(program, "displacementTex"), 0);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);

        glDisable(GL_DEPTH_TEST);
        glBindVertexArray(fullscreenVAO);

        // ============ PASS 2: Bright extract for bloom ============
        if (g_postProcess.enableBloom) {
            glBindFramebuffer(GL_FRAMEBUFFER, bloomFBO[0]);
            glViewport(0, 0, windowWidth / 2, windowHeight / 2);
            glUseProgram(brightExtractProg);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, gColorTex);
            glUniform1i(glGetUniformLocation(brightExtractProg, "sceneTex"), 0);
            glUniform1f(glGetUniformLocation(brightExtractProg, "threshold"), g_postProcess.bloomThreshold);
            glDrawArrays(GL_TRIANGLES, 0, 3);

            // ============ PASS 3: Gaussian blur (ping-pong) ============
            glUseProgram(blurProg);
            for (int i = 0; i < 4; i++) {  // 2 iterations (H+V each)
                bool horizontal = (i % 2 == 0);
                glBindFramebuffer(GL_FRAMEBUFFER, bloomFBO[horizontal ? 1 : 0]);
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, bloomTex[horizontal ? 0 : 1]);
                glUniform1i(glGetUniformLocation(blurProg, "inputTex"), 0);
                glUniform2f(glGetUniformLocation(blurProg, "direction"),
                    horizontal ? 1.0f : 0.0f, horizontal ? 0.0f : 1.0f);
                glUniform1f(glGetUniformLocation(blurProg, "texelSize"),
                    horizontal ? 2.0f / windowWidth : 2.0f / windowHeight);
                glDrawArrays(GL_TRIANGLES, 0, 3);
            }
        }

        // ============ PASS 4: SSAO ============
        if (g_postProcess.enableSSAO) {
            glBindFramebuffer(GL_FRAMEBUFFER, ssaoFBO);
            glViewport(0, 0, windowWidth, windowHeight);
            glUseProgram(ssaoProg);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, gDepthTex);
            glUniform1i(glGetUniformLocation(ssaoProg, "depthTex"), 0);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, gNormalTex);
            glUniform1i(glGetUniformLocation(ssaoProg, "normalTex"), 1);
            glUniformMatrix4fv(glGetUniformLocation(ssaoProg, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
            glUniform1f(glGetUniformLocation(ssaoProg, "radius"), g_postProcess.ssaoRadius);
            glDrawArrays(GL_TRIANGLES, 0, 3);
        }

        // ============ PASS 5: Composite to screen ============
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, windowWidth, windowHeight);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(compositeProg);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, gColorTex);
        glUniform1i(glGetUniformLocation(compositeProg, "sceneTex"), 0);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, bloomTex[0]);
        glUniform1i(glGetUniformLocation(compositeProg, "bloomTex"), 1);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, ssaoTex);
        glUniform1i(glGetUniformLocation(compositeProg, "aoTex"), 2);
        glUniform1f(glGetUniformLocation(compositeProg, "bloomIntensity"), g_postProcess.bloomIntensity);
        glUniform1i(glGetUniformLocation(compositeProg, "enableBloom"), g_postProcess.enableBloom);
        glUniform1i(glGetUniformLocation(compositeProg, "enableSSAO"), g_postProcess.enableSSAO);
        glUniform1i(glGetUniformLocation(compositeProg, "enableFXAA"), g_postProcess.enableFXAA);
        glUniform2f(glGetUniformLocation(compositeProg, "texelSize"), 1.0f / windowWidth, 1.0f / windowHeight);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        glBindVertexArray(0);

        // ImGui overlay
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Transparent overlay window in top-left corner
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);
        ImGui::SetNextWindowBgAlpha(0.5f);
        ImGui::Begin("Parameters", nullptr,
            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar);
        ImGui::Text("FPS: %.1f", io.Framerate);
        ImGui::Separator();
        ImGui::Text("Frequency: %.1f Hz", g_params.freq);
        ImGui::Text("Amplitude: %.1f", g_params.amp);
        ImGui::Text("Wave Speed: %.2f", g_params.c);
        ImGui::Separator();
        if (ImGui::Button("Clear Simulation (C)")) {
            g_clearSim = true;
        }

        // Post-processing controls
        ImGui::Separator();
        ImGui::Text("Post-Processing");
        ImGui::Checkbox("Bloom", &g_postProcess.enableBloom);
        if (g_postProcess.enableBloom) {
            ImGui::SliderFloat("Bloom Threshold", &g_postProcess.bloomThreshold, 0.1f, 2.0f);
            ImGui::SliderFloat("Bloom Intensity", &g_postProcess.bloomIntensity, 0.0f, 1.0f);
        }
        ImGui::Checkbox("SSAO", &g_postProcess.enableSSAO);
        if (g_postProcess.enableSSAO) {
            ImGui::SliderFloat("SSAO Radius", &g_postProcess.ssaoRadius, 0.1f, 2.0f);
        }
        ImGui::Checkbox("FXAA", &g_postProcess.enableFXAA);

        ImGui::Separator();
        ImGui::TextDisabled("Up/Down: Freq | Left/Right: Amp");
        ImGui::TextDisabled("W/S: Speed | R: Reset | C: Clear");
        ImGui::End();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Swap front and back buffers
        glfwSwapBuffers(window);
    }

    // Cleanup ImGui
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    // Cleanup
    cudaFree(device_prev_u);
    cudaFree(device_u);
    cudaFree(device_next_u);
    glDeleteTextures(1, &displacementTex);
    cudaGraphicsUnregisterResource(cudaResource);
    glDeleteProgram(program);
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);

    // Terminate GLFW
    glfwTerminate();

    return 0;
}
