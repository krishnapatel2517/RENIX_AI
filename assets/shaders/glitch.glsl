#version 330 core

uniform sampler2D screenTexture;
uniform float time;
uniform float glitchStrength;
uniform float glitchSpeed;
uniform float scanlineStrength;

in vec2 TexCoord;
out vec4 FragColor;

float random(vec2 coordinate)
{
    return fract(
        sin(
            dot(
                coordinate,
                vec2(12.9898, 78.233)
            )
        ) * 43758.5453
    );
}

void main()
{
    vec2 uv = TexCoord;

    // Time-based glitch trigger
    float glitchWave = sin(time * glitchSpeed);

    // Random horizontal distortion
    float block = floor(uv.y * 40.0);

    float noise = random(
        vec2(block, floor(time * glitchSpeed))
    );

    float distortion = 0.0;

    if (noise > 0.82)
    {
        distortion =
            (random(vec2(block, time)) - 0.5)
            * glitchStrength;
    }

    uv.x += distortion;

    // RGB channel separation
    float chromaticOffset =
        glitchStrength * 0.35;

    float red = texture(
        screenTexture,
        uv + vec2(chromaticOffset, 0.0)
    ).r;

    float green = texture(
        screenTexture,
        uv
    ).g;

    float blue = texture(
        screenTexture,
        uv - vec2(chromaticOffset, 0.0)
    ).b;

    vec4 original =
        texture(screenTexture, uv);

    vec3 glitchColor = vec3(
        red,
        green,
        blue
    );

    // Horizontal scanlines
    float scanline =
        sin(uv.y * 1200.0);

    scanline =
        scanline * 0.5 + 0.5;

    float scanEffect =
        1.0 - scanline * scanlineStrength;

    glitchColor *= scanEffect;

    // Subtle holographic green tint
    vec3 holoTint = vec3(
        0.05,
        1.0,
        0.65
    );

    glitchColor =
        mix(
            glitchColor,
            glitchColor * holoTint * 1.2,
            0.08
        );

    // Blend original and distorted image
    float blendAmount =
        clamp(
            abs(glitchWave) * glitchStrength * 2.0,
            0.0,
            1.0
        );

    vec3 finalColor =
        mix(
            original.rgb,
            glitchColor,
            blendAmount
        );

    FragColor = vec4(
        finalColor,
        original.a
    );
}