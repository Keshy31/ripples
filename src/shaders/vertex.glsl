#version 450
layout (location = 0) in vec3 in_pos;
out vec2 uv;
out vec3 worldPos;
out vec3 normal;

uniform sampler2D displacementTex;
uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

void main() {
    uv = in_pos.xz * 0.5 + 0.5; // Assuming xz plane
    float h = texture(displacementTex, uv).r;

    vec3 displacedPos = in_pos + vec3(0.0, h * 0.1, 0.0);
    worldPos = (model * vec4(displacedPos, 1.0)).xyz;
    gl_Position = projection * view * vec4(worldPos, 1.0);

    // Simple normal calculation
    float h_L = texture(displacementTex, uv - vec2(1.0/2048.0, 0.0)).r;
    float h_R = texture(displacementTex, uv + vec2(1.0/2048.0, 0.0)).r;
    float h_D = texture(displacementTex, uv - vec2(0.0, 1.0/2048.0)).r;
    float h_U = texture(displacementTex, uv + vec2(0.0, 1.0/2048.0)).r;

    vec3 N = normalize(vec3(h_L - h_R, 0.1, h_D - h_U));
    normal = normalize(mat3(transpose(inverse(model))) * N);
}
