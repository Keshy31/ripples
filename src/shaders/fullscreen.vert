#version 450
out vec2 texCoord;

void main() {
    // Fullscreen triangle: vertices at (-1,-1), (3,-1), (-1,3)
    // This oversized triangle covers the entire screen
    float x = (gl_VertexID == 1) ? 3.0 : -1.0;
    float y = (gl_VertexID == 2) ? 3.0 : -1.0;
    gl_Position = vec4(x, y, 0.0, 1.0);
    texCoord = vec2(x, y) * 0.5 + 0.5;
}
