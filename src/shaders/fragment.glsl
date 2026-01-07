#version 450
in vec2 uv;
in vec3 worldPos;
in vec3 normal;
in vec3 viewNormal;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 fragNormal;

uniform sampler2D displacementTex;
uniform vec3 camPos;
uniform vec3 lightPos;
uniform vec3 lightColor = vec3(1.0);

void main() {
    vec3 N = normalize(normal);
    vec3 V = normalize(camPos - worldPos);
    vec3 L = normalize(lightPos - worldPos);
    vec3 H = normalize(L + V);

    // Get height for coloring
    float h = texture(displacementTex, uv).r;
    float heightNorm = clamp(h * 0.15 + 0.5, 0.0, 1.0);

    // Vibrant color gradient: deep purple -> cyan -> bright white
    vec3 colorLow = vec3(0.1, 0.0, 0.2);    // Deep purple
    vec3 colorMid = vec3(0.0, 0.6, 0.8);    // Cyan
    vec3 colorHigh = vec3(1.0, 1.0, 1.0);   // White

    vec3 baseColor;
    if (heightNorm < 0.5) {
        baseColor = mix(colorLow, colorMid, heightNorm * 2.0);
    } else {
        baseColor = mix(colorMid, colorHigh, (heightNorm - 0.5) * 2.0);
    }

    // Lighting with specular
    float ambient = 0.2;
    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), 64.0);

    vec3 color = baseColor * (ambient + diff * 0.6) + vec3(1.0) * spec * 0.4;

    fragColor = vec4(color, 1.0);

    // Output view-space normal for SSAO (encoded to 0-1 range)
    fragNormal = vec4(normalize(viewNormal) * 0.5 + 0.5, 1.0);
}
