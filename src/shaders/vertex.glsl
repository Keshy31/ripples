#version 450
layout (location = 0) in vec3 in_pos;
out vec2 uv;
out vec3 worldPos;
out vec3 normal;
out vec3 viewNormal;

uniform sampler2D displacementTex;
uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

void main() {
    uv = in_pos.xz * 0.5 + 0.5; // Assuming xz plane
    float h = texture(displacementTex, uv).r;

    float dispScale = 0.05;  // Displacement scale factor
    vec3 displacedPos = in_pos + vec3(0.0, h * dispScale, 0.0);
    worldPos = (model * vec4(displacedPos, 1.0)).xyz;
    gl_Position = projection * view * vec4(worldPos, 1.0);

    // Normal calculation from heightfield gradients
    float texelSize = 1.0 / 2048.0;
    float h_L = texture(displacementTex, uv - vec2(texelSize, 0.0)).r;
    float h_R = texture(displacementTex, uv + vec2(texelSize, 0.0)).r;
    float h_D = texture(displacementTex, uv - vec2(0.0, texelSize)).r;
    float h_U = texture(displacementTex, uv + vec2(0.0, texelSize)).r;

    // Scale gradients to match displacement scale
    float normalStrength = dispScale * 50.0;
    vec3 N = normalize(vec3((h_L - h_R) * normalStrength, 1.0, (h_D - h_U) * normalStrength));
    normal = normalize(mat3(transpose(inverse(model))) * N);

    // View-space normal for SSAO
    mat3 normalMatrix = mat3(transpose(inverse(view * model)));
    viewNormal = normalMatrix * N;
}
