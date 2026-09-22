#version 330 core

uniform sampler2D screenTexture;
uniform float glowStrength;
uniform float glowRadius;
uniform vec2 texelSize;

in vec2 TexCoord;
out vec4 FragColor;

void main()
{
    vec4 baseColor = texture(screenTexture, TexCoord);

    vec4 glowColor = vec4(0.0);

    // Multi-directional blur samples
    glowColor += texture(
        screenTexture,
        TexCoord + vec2(texelSize.x * glowRadius, 0.0)
    );

    glowColor += texture(
        screenTexture,
        TexCoord - vec2(texelSize.x * glowRadius, 0.0)
    );

    glowColor += texture(
        screenTexture,
        TexCoord + vec2(0.0, texelSize.y * glowRadius)
    );

    glowColor += texture(
        screenTexture,
        TexCoord - vec2(0.0, texelSize.y * glowRadius)
    );

    glowColor += texture(
        screenTexture,
        TexCoord + vec2(
            texelSize.x * glowRadius,
            texelSize.y * glowRadius
        )
    );

    glowColor += texture(
        screenTexture,
        TexCoord + vec2(
            -texelSize.x * glowRadius,
            texelSize.y * glowRadius
        )
    );

    glowColor += texture(
        screenTexture,
        TexCoord + vec2(
            texelSize.x * glowRadius,
            -texelSize.y * glowRadius
        )
    );

    glowColor += texture(
        screenTexture,
        TexCoord - vec2(
            texelSize.x * glowRadius,
            texelSize.y * glowRadius
        )
    );

    // Average the samples
    glowColor /= 8.0;

    // Holographic green glow
    vec3 holoGlow = vec3(
        0.05,
        1.0,
        0.65
    );

    float intensity = glowColor.a * glowStrength;

    vec3 finalColor =
        baseColor.rgb +
        holoGlow * intensity;

    float finalAlpha = max(
        baseColor.a,
        glowColor.a * glowStrength
    );

    FragColor = vec4(
        finalColor,
        finalAlpha
    );
}