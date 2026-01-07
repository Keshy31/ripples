#version 450
in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D inputTex;
uniform vec2 direction; // (1,0) for horizontal, (0,1) for vertical
uniform float texelSize; // 1.0 / texture_width or height

// 9-tap Gaussian weights (sigma ~= 2.0)
const float weights[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

void main() {
    vec3 result = texture(inputTex, texCoord).rgb * weights[0];

    for (int i = 1; i < 5; i++) {
        vec2 offset = direction * texelSize * float(i);
        result += texture(inputTex, texCoord + offset).rgb * weights[i];
        result += texture(inputTex, texCoord - offset).rgb * weights[i];
    }

    fragColor = vec4(result, 1.0);
}
