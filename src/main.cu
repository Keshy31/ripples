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

// Forward declaration of CUDA kernel
extern "C" __global__ void fused_update_kernel(float* prev_u, float* u, float* next_u, float c, float dt, float dx, float damping, float freq, float amp, float t, int source_x, int source_y, int size);

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

    // Initialize GLAD
    if (!gladLoadGL(glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // CUDA setup
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    std::cout << "Using GPU: " << deviceProp.name << std::endl;
    
    // Simulation constants
    const size_t grid_size = 2048;
    const float dx = 1.0f / static_cast<float>(grid_size - 1);
    const float c = 0.5f;  // Wave speed (increased for faster propagation)
    const float dt = dx / (c * 1.5f);
    const float damping = 0.01f;
    float t = 0.0f;
    float freq = 10.0f; // Example frequency
    float amp = 5.0f;   // Wave amplitude (increased for visibility)
    
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
    
    GLuint vertexShader = loadShader("src/shaders/vertex.glsl", GL_VERTEX_SHADER);
    GLuint fragmentShader = loadShader("src/shaders/fragment.glsl", GL_FRAGMENT_SHADER);
    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    int linkSuccess;
    char linkInfoLog[512];
    glGetProgramiv(program, GL_LINK_STATUS, &linkSuccess);
    if (!linkSuccess) {
        glGetProgramInfoLog(program, 512, NULL, linkInfoLog);
        std::cerr << "Program link error: " << linkInfoLog << std::endl;
    }
    
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
    
    // Matrices
    glm::mat4 projection = glm::perspective(glm::radians(60.0f), (float)windowWidth / (float)windowHeight, 0.1f, 100.0f);
    glm::vec3 camPos(0.0f, 5.0f, 0.001f);  // Top-down view (slight z offset to avoid gimbal lock)
    glm::mat4 view = glm::lookAt(camPos, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f));
    glm::mat4 model = glm::scale(glm::mat4(1.0f), glm::vec3(4.0f, 1.0f, 4.0f));  // Scale mesh 4x to fill view
    glm::vec3 lightPos(2.0f, 5.0f, 2.0f);  // Offset light for better shading
    
    // Set the clear color
    glClearColor(0.02f, 0.02f, 0.05f, 1.0f);  // Dark background

    // After glad init
    glEnable(GL_DEPTH_TEST);

    // Loop until the user closes the window
    while (!glfwWindowShouldClose(window)) {
        // Simulate step
        dim3 block_dim(16, 16, 1);
        dim3 grid_dim((grid_size + 15) / 16, (grid_size + 15) / 16, 1);
        fused_update_kernel<<<grid_dim, block_dim>>>(device_prev_u, device_u, device_next_u, c, dt, dx, damping, freq, amp, t, grid_size / 2, grid_size / 2, grid_size);
        cudaDeviceSynchronize();
        t += dt;
    
        // Swap pointers
        float* temp = device_prev_u;
        device_prev_u = device_u;
        device_u = device_next_u;
        device_next_u = temp;
    
        // Update texture with CUDA
        cudaGraphicsMapResources(1, &cudaResource);
        cudaArray_t array;
        cudaGraphicsSubResourceGetMappedArray(&array, cudaResource, 0, 0);
        cudaMemcpy2DToArray(array, 0, 0, device_u, grid_size * sizeof(float), grid_size * sizeof(float), grid_size, cudaMemcpyDeviceToDevice);
        cudaGraphicsUnmapResources(1, &cudaResource);
    
        // Poll for and process events
        glfwPollEvents();

        // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Render
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
        // Set other uniforms...
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
    
        // Swap front and back buffers
        glfwSwapBuffers(window);
    }

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
