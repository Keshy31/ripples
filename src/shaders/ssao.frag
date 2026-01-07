#version 450
in vec2 texCoord;
out float fragColor;

uniform sampler2D depthTex;
uniform sampler2D normalTex;
uniform mat4 projection;
uniform vec2 noiseScale; // screen dimensions / noise texture size
uniform float radius = 0.5;
uniform float bias = 0.025;

// Reconstruct view-space position from depth
vec3 getViewPos(vec2 uv, float depth) {
    vec4 clipPos = vec4(uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    vec4 viewPos = inverse(projection) * clipPos;
    return viewPos.xyz / viewPos.w;
}

// Simple hash for random sampling
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    float depth = texture(depthTex, texCoord).r;
    if (depth >= 1.0) {
        fragColor = 1.0;
        return;
    }

    vec3 viewPos = getViewPos(texCoord, depth);
    vec3 normal = texture(normalTex, texCoord).rgb * 2.0 - 1.0;

    // Random rotation per pixel
    float randomAngle = hash(texCoord * 1000.0) * 6.28318;
    vec3 randomVec = vec3(cos(randomAngle), sin(randomAngle), 0.0);

    // Create TBN matrix
    vec3 tangent = normalize(randomVec - normal * dot(randomVec, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 TBN = mat3(tangent, bitangent, normal);

    // Sample hemisphere
    float occlusion = 0.0;
    const int samples = 16;

    for (int i = 0; i < samples; i++) {
        // Hemisphere sample positions (precomputed would be better)
        float fi = float(i);
        float r = hash(vec2(fi, texCoord.x)) * radius;
        float theta = hash(vec2(fi, texCoord.y)) * 6.28318;
        float phi = hash(vec2(fi, fi)) * 1.5708;

        vec3 sampleDir = vec3(
            sin(phi) * cos(theta),
            sin(phi) * sin(theta),
            cos(phi)
        );

        vec3 samplePos = viewPos + TBN * sampleDir * r;

        // Project sample to screen space
        vec4 offset = projection * vec4(samplePos, 1.0);
        offset.xyz /= offset.w;
        offset.xyz = offset.xyz * 0.5 + 0.5;

        float sampleDepth = texture(depthTex, offset.xy).r;
        vec3 sampleViewPos = getViewPos(offset.xy, sampleDepth);

        // Range check and occlusion
        float rangeCheck = smoothstep(0.0, 1.0, radius / abs(viewPos.z - sampleViewPos.z));
        occlusion += (sampleViewPos.z >= samplePos.z + bias ? 1.0 : 0.0) * rangeCheck;
    }

    fragColor = 1.0 - (occlusion / float(samples));
}
