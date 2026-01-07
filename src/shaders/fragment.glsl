#version 450
in vec2 uv;
in vec3 worldPos;
in vec3 normal;
out vec4 fragColor;

uniform vec3 camPos;
uniform vec3 lightPos;
uniform vec3 lightColor = vec3(1.0);

void main() {
    vec3 N = normalize(normal);
    vec3 V = normalize(camPos - worldPos);
    vec3 L = normalize(lightPos - worldPos);
    
    float diff = max(dot(N, L), 0.0);
    vec3 diffuse = diff * lightColor;
    
    fragColor = vec4(diffuse, 1.0);
}
