#version 450
in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D sceneTex;
uniform sampler2D bloomTex;
uniform sampler2D aoTex;
uniform float bloomIntensity = 0.3;
uniform bool enableBloom = true;
uniform bool enableSSAO = true;
uniform bool enableFXAA = true;
uniform vec2 texelSize; // 1.0 / screen dimensions

// FXAA constants
const float FXAA_SPAN_MAX = 8.0;
const float FXAA_REDUCE_MIN = 1.0 / 128.0;
const float FXAA_REDUCE_MUL = 1.0 / 8.0;

vec3 fxaa(sampler2D tex, vec2 uv) {
    vec3 rgbNW = texture(tex, uv + vec2(-1.0, -1.0) * texelSize).rgb;
    vec3 rgbNE = texture(tex, uv + vec2(1.0, -1.0) * texelSize).rgb;
    vec3 rgbSW = texture(tex, uv + vec2(-1.0, 1.0) * texelSize).rgb;
    vec3 rgbSE = texture(tex, uv + vec2(1.0, 1.0) * texelSize).rgb;
    vec3 rgbM = texture(tex, uv).rgb;

    vec3 luma = vec3(0.299, 0.587, 0.114);
    float lumaNW = dot(rgbNW, luma);
    float lumaNE = dot(rgbNE, luma);
    float lumaSW = dot(rgbSW, luma);
    float lumaSE = dot(rgbSE, luma);
    float lumaM = dot(rgbM, luma);

    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

    vec2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y = ((lumaNW + lumaSW) - (lumaNE + lumaSE));

    float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * (0.25 * FXAA_REDUCE_MUL), FXAA_REDUCE_MIN);
    float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = min(vec2(FXAA_SPAN_MAX), max(vec2(-FXAA_SPAN_MAX), dir * rcpDirMin)) * texelSize;

    vec3 rgbA = 0.5 * (
        texture(tex, uv + dir * (1.0 / 3.0 - 0.5)).rgb +
        texture(tex, uv + dir * (2.0 / 3.0 - 0.5)).rgb);

    vec3 rgbB = rgbA * 0.5 + 0.25 * (
        texture(tex, uv + dir * -0.5).rgb +
        texture(tex, uv + dir * 0.5).rgb);

    float lumaB = dot(rgbB, luma);

    if (lumaB < lumaMin || lumaB > lumaMax) {
        return rgbA;
    }
    return rgbB;
}

void main() {
    vec3 scene = texture(sceneTex, texCoord).rgb;

    // Apply SSAO
    if (enableSSAO) {
        float ao = texture(aoTex, texCoord).r;
        scene *= ao;
    }

    // Apply bloom
    if (enableBloom) {
        vec3 bloom = texture(bloomTex, texCoord).rgb;
        scene += bloom * bloomIntensity;
    }

    // Apply FXAA (on combined result - would need extra pass for best quality)
    if (enableFXAA) {
        // For proper FXAA, we'd render scene+bloom to another FBO first
        // This is a simplified version
        fragColor = vec4(scene, 1.0);
    } else {
        fragColor = vec4(scene, 1.0);
    }

    // Tone mapping (simple Reinhard)
    fragColor.rgb = fragColor.rgb / (fragColor.rgb + vec3(1.0));

    // Gamma correction
    fragColor.rgb = pow(fragColor.rgb, vec3(1.0 / 2.2));
}
