#version 450
in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D sceneTex;
uniform float threshold = 0.8;

void main() {
    vec3 color = texture(sceneTex, texCoord).rgb;
    float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722));

    if (brightness > threshold) {
        fragColor = vec4(color * (brightness - threshold), 1.0);
    } else {
        fragColor = vec4(0.0);
    }
}
