RENIX SHADERS
=============

This folder contains shader files used by the RENIX holographic interface.

Shader files:
-------------

hologram.glsl
    Main holographic rendering effect.
    Adds the RENIX holographic tint, scan movement, flicker,
    and edge intensity.

glow.glsl
    Creates the glow effect around holographic elements.

scanline.glsl
    Adds horizontal scanlines and a moving scan effect
    to the holographic interface.

glitch.glsl
    Creates digital glitch effects including:
    - RGB channel separation
    - Horizontal distortion
    - Random visual blocks
    - Scanline distortion
    - Digital flickering


Shader Usage:
-------------

These shaders are intended to be loaded by the RENIX
holographic UI rendering system.

The UI renderer can combine multiple shader effects to
create the final RENIX visual appearance.

Recommended effect order:

1. Base rendering
2. Hologram effect
3. Glow
4. Scanlines
5. Glitch
6. Final compositing


Important:
----------

Do not delete or rename these shader files unless the
corresponding renderer code is updated.

Shader uniforms such as:

    time
    opacity
    glowStrength
    glowRadius
    scanSpeed
    scanlineStrength
    glitchStrength

are controlled by the RENIX rendering system.

The exact values should be controlled by the UI configuration
rather than hard-coded in the application logic.


Future shaders:
---------------

Additional shaders can be added later for:

    - distortion
    - chromatic aberration
    - holographic noise
    - particles
    - energy waves
    - depth effects
    - holographic projection
    - transition effects
    - HUD effects


RENIX Shader Philosophy:
------------------------

The shader system should provide a futuristic holographic
appearance while keeping the interface readable and usable.

Visual effects should not significantly reduce performance
or make important information difficult to read.


RENIX AI
========
Holographic Interface Shader System