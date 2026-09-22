#version 330 core

uniform sampler2D screenTexture;

uniform float time;
uniform float scanSpeed;
uniform float scanlineStrength;
uniform float lineSpacing;
uniform float lineThickness;
uniform float flickerStrength;

in vec2 TexCoord;

out vec4 FragColor;

void main()
{
    vec4 baseColor = texture(
        screenTexture,
        TexCoord
    );

    // Moving scanline position
    float movingLine =
        fract(
            TexCoord.y * lineSpacing
            - time * scanSpeed
        );

    // Thin bright scanline
    float scanline =
        smoothstep(
            lineThickness,
            0.0,
            abs(movingLine - 0.5)
        );

    // Static horizontal lines
    float staticLines =
        sin(
            TexCoord.y *
            lineSpacing *
            6.28318530718
        );

    staticLines =
        staticLines * 0.5 + 0.5;

    // Combine moving and static scanlines
    float scanEffect =
        scanline * scanlineStrength;

    scanEffect +=
        staticLines *
        scanlineStrength *
        0.15;

    // Subtle flicker
    float flicker =
        1.0 +
        sin(time * 7.0) *
        flickerStrength;

    // RENIX holographic green
    vec3 holoColor =
        vec3(
            0.05,
            1.0,
            0.65
        );

    vec3 scanColor =
        holoColor *
        scanEffect *
        flicker;

    vec3 finalColor =
        baseColor.rgb +
        scanColor;

    FragColor =
        vec4(
            finalColor,
            baseColor.a
        );
}