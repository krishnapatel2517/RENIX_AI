#version 330 core

uniform sampler2D screenTexture;
uniform float time;
uniform float opacity;
uniform float scanSpeed;
uniform float glowStrength;

in vec2 TexCoord;
out vec4 FragColor;

void main()
{
    vec4 color = texture(screenTexture, TexCoord);

    // Horizontal scanline movement
    float scan = sin((TexCoord.y * 900.0) - (time * scanSpeed));
    scan = scan * 0.5 + 0.5;

    // Subtle holographic flicker
    float flicker = 0.96 + 0.04 * sin(time * 8.0);

    // Edge detection approximation
    float edgeX = abs(TexCoord.x - 0.5) * 2.0;
    float edgeY = abs(TexCoord.y - 0.5) * 2.0;
    float edge = max(edgeX, edgeY);

    // Holographic intensity
    float hologram = 0.75 + scan * 0.25;

    vec3 holoColor = vec3(
        0.05,
        1.0,
        0.65
    );

    vec3 finalColor =
        color.rgb *
        holoColor *
        hologram *
        flicker;

    // Slight edge glow
    finalColor += holoColor * edge * glowStrength * 0.15;

    FragColor = vec4(
        finalColor,
        color.a * opacity
    );
}