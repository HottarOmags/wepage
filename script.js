"use strict";

const canvas = document.getElementById('geometric-canvas');
const gl = canvas.getContext('webgl');

if (!gl) {
    console.error('WebGL not supported!');
}

canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

// --- WebGL Utility Functions ---
function createShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error('Shader compilation error:', gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        return null;
    }
    return shader;
}

function createProgram(gl, vertexShader, fragmentShader) {
    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        console.error('Program linking error:', gl.getProgramInfoLog(program));
        gl.deleteProgram(program);
        return null;
    }
    return program;
}

// --- Full-screen Quad (for pixel-based effects) ---
const quadVS = `
    attribute vec2 a_position;
    void main() {
        gl_Position = vec4(a_position, 0, 1);
    }
`;

const quadPositions = new Float32Array([
    -1.0, -1.0,
     1.0, -1.0,
    -1.0,  1.0,
    -1.0,  1.0,
     1.0, -1.0,
     1.0,  1.0,
]);

const quadBuffer = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
gl.bufferData(gl.ARRAY_BUFFER, quadPositions, gl.STATIC_DRAW);

// --- Effect Management ---
let currentEffectIndex = 0;
const effects = [];
let startTime = performance.now();
const effectDuration = 5000; // 5 seconds per effect

function nextEffect() {
    if (!effects.length) {
        console.warn('No effects available to switch.');
        return;
    }
    currentEffectIndex = (currentEffectIndex + 1) % effects.length;
    const eff = effects[currentEffectIndex];
    if (eff && typeof eff.init === 'function') {
        eff.init(); // Re-initialize the new effect
    }
    startTime = performance.now(); // Reset timer on manual switch
    updateEffectIndicator(); // Update the indicator
    console.log('Switched to effect:', currentEffectIndex);
}

// --- Effect 1: Starfield ---
const starfield = {};
starfield.vsSource = `
    attribute vec2 a_position;
    attribute float a_size;
    uniform vec2 u_resolution;
    uniform float u_time;

    void main() {
        vec2 clipSpace = ((a_position / u_resolution) * 2.0 - 1.0) * vec2(1, -1);
        gl_Position = vec4(clipSpace, 0, 1);
        gl_PointSize = a_size;
    }
`;

starfield.fsSource = `
    precision mediump float;
    void main() {
        gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0); // White stars
    }
`;

starfield.init = () => {
    starfield.program = createProgram(gl,
        createShader(gl, gl.VERTEX_SHADER, starfield.vsSource),
        createShader(gl, gl.FRAGMENT_SHADER, starfield.fsSource)
    );
    gl.useProgram(starfield.program);

    starfield.positionAttributeLocation = gl.getAttribLocation(starfield.program, 'a_position');
    starfield.sizeAttributeLocation = gl.getAttribLocation(starfield.program, 'a_size');
    starfield.resolutionUniformLocation = gl.getUniformLocation(starfield.program, 'u_resolution');
    starfield.timeUniformLocation = gl.getUniformLocation(starfield.program, 'u_time');

    const numStars = 1000;
    const starData = []; // x, y, size, speed
    for (let i = 0; i < numStars; i++) {
        starData.push(Math.random() * canvas.width);
        starData.push(Math.random() * canvas.height);
        starData.push(Math.random() * 3 + 1); // size
        starData.push(Math.random() * 2 + 0.5); // speed
    }
    starfield.starBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, starfield.starBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(starData), gl.DYNAMIC_DRAW);
    starfield.numStars = numStars;
    starfield.starData = starData; // Keep reference to update
};

starfield.draw = (time) => {
    gl.useProgram(starfield.program);
    gl.uniform2f(starfield.resolutionUniformLocation, canvas.width, canvas.height);
    gl.uniform1f(starfield.timeUniformLocation, time);

    // Update star positions on CPU and re-upload (simple approach)
    for (let i = 0; i < starfield.numStars; i++) {
        const index = i * 4;
        starfield.starData[index] -= starfield.starData[index + 3]; // x -= speed
        if (starfield.starData[index] < 0) {
            starfield.starData[index] = canvas.width;
            starfield.starData[index + 1] = Math.random() * canvas.height;
        }
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, starfield.starBuffer);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, new Float32Array(starfield.starData));

    gl.enableVertexAttribArray(starfield.positionAttributeLocation);
    gl.vertexAttribPointer(starfield.positionAttributeLocation, 2, gl.FLOAT, false, 4 * 4, 0);

    gl.enableVertexAttribArray(starfield.sizeAttributeLocation);
    gl.vertexAttribPointer(starfield.sizeAttributeLocation, 1, gl.FLOAT, false, 4 * 4, 2 * 4);

    gl.drawArrays(gl.POINTS, 0, starfield.numStars);
};
effects.push(starfield);

// --- Effect 2: Plasma ---
const plasma = {};
plasma.fsSource = `
    precision highp float;
    uniform vec2 u_resolution;
    uniform float u_time;

    vec3 palette(float t) {
        vec3 a = vec3(0.5, 0.5, 0.5);
        vec3 b = vec3(0.5, 0.5, 0.5);
        vec3 c = vec3(1.0, 1.0, 1.0);
        vec3 d = vec3(0.263, 0.416, 0.557);
        return a + b * cos(6.28318 * (c * t + d));
    }

    void main() {
        vec2 uv = gl_FragCoord.xy / u_resolution;
        float x = uv.x * 10.0;
        float y = uv.y * 10.0;

        float color = sin(x + u_time) +
                      sin(y + u_time / 2.0) +
                      sin((x + y) / 2.0 + u_time / 3.0) +
                      sin(sqrt(x * x + y * y) + u_time / 4.0);
        color = color / 4.0 + 0.5; // Normalize to 0-1

        gl_FragColor = vec4(palette(color), 1.0);
    }
`;

plasma.init = () => {
    plasma.program = createProgram(gl,
        createShader(gl, gl.VERTEX_SHADER, quadVS),
        createShader(gl, gl.FRAGMENT_SHADER, plasma.fsSource)
    );
    gl.useProgram(plasma.program);

    plasma.positionAttributeLocation = gl.getAttribLocation(plasma.program, 'a_position');
    plasma.resolutionUniformLocation = gl.getUniformLocation(plasma.program, 'u_resolution');
    plasma.timeUniformLocation = gl.getUniformLocation(plasma.program, 'u_time');
};

plasma.draw = (time) => {
    gl.useProgram(plasma.program);
    gl.uniform2f(plasma.resolutionUniformLocation, canvas.width, canvas.height);
    gl.uniform1f(plasma.timeUniformLocation, time / 1000.0); // Convert ms to seconds

    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.enableVertexAttribArray(plasma.positionAttributeLocation);
    gl.vertexAttribPointer(plasma.positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
};
effects.push(plasma);

// --- Effect 3: Sine Wave ---
const sineWave = {};
sineWave.vsSource = quadVS; // Use the same vertex shader for full-screen quad
sineWave.fsSource = `
    precision highp float;
    uniform vec2 u_resolution;
    uniform float u_time;

    void main() {
        vec2 uv = gl_FragCoord.xy / u_resolution;
        float y_offset = sin(uv.x * 20.0 + u_time * 2.0) * 0.1; // Adjust amplitude and frequency
        float thickness = 0.01; // Thickness of the wave line

        float dist = abs(uv.y - (0.5 + y_offset)); // Distance from the wave center
        
        vec3 color = vec3(0.0); // Black background
        if (dist < thickness) {
            color = vec3(0.0, 1.0, 1.0); // Cyan wave
        }
        
        gl_FragColor = vec4(color, 1.0);
    }
`;

sineWave.init = () => {
    sineWave.program = createProgram(gl,
        createShader(gl, gl.VERTEX_SHADER, sineWave.vsSource),
        createShader(gl, gl.FRAGMENT_SHADER, sineWave.fsSource)
    );
    gl.useProgram(sineWave.program);

    sineWave.positionAttributeLocation = gl.getAttribLocation(sineWave.program, 'a_position');
    sineWave.resolutionUniformLocation = gl.getUniformLocation(sineWave.program, 'u_resolution');
    sineWave.timeUniformLocation = gl.getUniformLocation(sineWave.program, 'u_time');
};

sineWave.draw = (time) => {
    gl.useProgram(sineWave.program);
    gl.uniform2f(sineWave.resolutionUniformLocation, canvas.width, canvas.height);
    gl.uniform1f(sineWave.timeUniformLocation, time / 1000.0);

    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.enableVertexAttribArray(sineWave.positionAttributeLocation);
    gl.vertexAttribPointer(sineWave.positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
};
effects.push(sineWave);

// --- Effect 4: Color Cycle (Simple Gradient Shift) ---
const colorCycle = {};
colorCycle.vsSource = quadVS;
colorCycle.fsSource = `
    precision highp float;
    uniform vec2 u_resolution;
    uniform float u_time;

    void main() {
        vec2 uv = gl_FragCoord.xy / u_resolution;
        
        // Simple color cycling based on time and position
        float r = sin(uv.x * 10.0 + u_time * 0.5) * 0.5 + 0.5;
        float g = sin(uv.y * 10.0 + u_time * 0.7) * 0.5 + 0.5;
        float b = sin((uv.x + uv.y) * 5.0 + u_time * 0.9) * 0.5 + 0.5;
        
        gl_FragColor = vec4(r, g, b, 1.0);
    }
`;

colorCycle.init = () => {
    colorCycle.program = createProgram(gl,
        createShader(gl, gl.VERTEX_SHADER, colorCycle.vsSource),
        createShader(gl, gl.FRAGMENT_SHADER, colorCycle.fsSource)
    );
    gl.useProgram(colorCycle.program);

    colorCycle.positionAttributeLocation = gl.getAttribLocation(colorCycle.program, 'a_position');
    colorCycle.resolutionUniformLocation = gl.getUniformLocation(colorCycle.program, 'u_resolution');
    colorCycle.timeUniformLocation = gl.getUniformLocation(colorCycle.program, 'u_time');
};

colorCycle.draw = (time) => {
    gl.useProgram(colorCycle.program);
    gl.uniform2f(colorCycle.resolutionUniformLocation, canvas.width, canvas.height);
    gl.uniform1f(colorCycle.timeUniformLocation, time / 1000.0);

    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.enableVertexAttribArray(colorCycle.positionAttributeLocation);
    gl.vertexAttribPointer(colorCycle.positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
};
effects.push(colorCycle);

// --- Effect 5: Tunnel Effect ---
const tunnel = {};
tunnel.vsSource = quadVS;
tunnel.fsSource = `
    precision highp float;
    uniform vec2 u_resolution;
    uniform float u_time;

    void main() {
        vec2 uv = (gl_FragCoord.xy / u_resolution) * 2.0 - 1.0;
        uv.x *= u_resolution.x / u_resolution.y; // Aspect ratio correction

        float angle = atan(uv.y, uv.x);
        float radius = length(uv);

        float speed = u_time * 0.5;
        float x = radius * cos(angle + speed);
        float y = radius * sin(angle + speed);

        float color_r = sin(x * 10.0) * 0.5 + 0.5;
        float color_g = cos(y * 10.0) * 0.5 + 0.5;
        float color_b = sin((x + y) * 5.0) * 0.5 + 0.5;

        gl_FragColor = vec4(color_r, color_g, color_b, 1.0);
    }
`;

tunnel.init = () => {
    tunnel.program = createProgram(gl,
        createShader(gl, gl.VERTEX_SHADER, tunnel.vsSource),
        createShader(gl, gl.FRAGMENT_SHADER, tunnel.fsSource)
    );
    gl.useProgram(tunnel.program);

    tunnel.positionAttributeLocation = gl.getAttribLocation(tunnel.program, 'a_position');
    tunnel.resolutionUniformLocation = gl.getUniformLocation(tunnel.program, 'u_resolution');
    tunnel.timeUniformLocation = gl.getUniformLocation(tunnel.program, 'u_time');
};

tunnel.draw = (time) => {
    gl.useProgram(tunnel.program);
    gl.uniform2f(tunnel.resolutionUniformLocation, canvas.width, canvas.height);
    gl.uniform1f(tunnel.timeUniformLocation, time / 1000.0);

    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.enableVertexAttribArray(tunnel.positionAttributeLocation);
    gl.vertexAttribPointer(tunnel.positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
};
effects.push(tunnel);

// --- Effect 7: Metaballs ---
const metaballs = {};
metaballs.vsSource = quadVS;
metaballs.fsSource = `
    precision highp float;
    uniform vec2 u_resolution;
    uniform float u_time;

    const int NUM_BALLS = 4;
    uniform vec2 u_ballPositions[NUM_BALLS];
    uniform float u_ballRadii[NUM_BALLS];

    void main() {
        vec2 uv = gl_FragCoord.xy / u_resolution;
        float value = 0.0;

        for (int i = 0; i < NUM_BALLS; i++) {
            float dist = distance(uv, u_ballPositions[i]);
            value += u_ballRadii[i] / dist;
        }

        vec3 color = vec3(0.0);
        if (value > 3.0) {
            color = vec3(0.0, 0.5, 1.0); // Blue
        } else if (value > 2.0) {
            color = vec3(0.0, 0.7, 1.0); // Lighter Blue
        } else if (value > 1.0) {
            color = vec3(0.0, 0.9, 1.0); // Even Lighter Blue
        }

        gl_FragColor = vec4(color, 1.0);
    }
`;

metaballs.init = () => {
    metaballs.program = createProgram(gl,
        createShader(gl, gl.VERTEX_SHADER, metaballs.vsSource),
        createShader(gl, gl.FRAGMENT_SHADER, metaballs.fsSource)
    );
    gl.useProgram(metaballs.program);

    metaballs.positionAttributeLocation = gl.getAttribLocation(metaballs.program, 'a_position');
    metaballs.resolutionUniformLocation = gl.getUniformLocation(metaballs.program, 'u_resolution');
    metaballs.timeUniformLocation = gl.getUniformLocation(metaballs.program, 'u_time');
    metaballs.ballPositionsUniformLocation = gl.getUniformLocation(metaballs.program, 'u_ballPositions');
    metaballs.ballRadiiUniformLocation = gl.getUniformLocation(metaballs.program, 'u_ballRadii');

    metaballs.ballData = [];
    for (let i = 0; i < 4; i++) {
        metaballs.ballData.push({
            x: Math.random(),
            y: Math.random(),
            radius: Math.random() * 0.1 + 0.05,
            speedX: (Math.random() - 0.5) * 0.001,
            speedY: (Math.random() - 0.5) * 0.001
        });
    }
};

metaballs.draw = (time) => {
    gl.useProgram(metaballs.program);
    gl.uniform2f(metaballs.resolutionUniformLocation, canvas.width, canvas.height);
    gl.uniform1f(metaballs.timeUniformLocation, time / 1000.0);

    const positions = [];
    const radii = [];
    for (let i = 0; i < metaballs.ballData.length; i++) {
        const ball = metaballs.ballData[i];
        ball.x += ball.speedX;
        ball.y += ball.speedY;

        if (ball.x < 0 || ball.x > 1) ball.speedX *= -1;
        if (ball.y < 0 || ball.y > 1) ball.speedY *= -1;

        positions.push(ball.x, ball.y);
        radii.push(ball.radius);
    }
    gl.uniform2fv(metaballs.ballPositionsUniformLocation, new Float32Array(positions));
    gl.uniform1fv(metaballs.ballRadiiUniformLocation, new Float32Array(radii));

    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.enableVertexAttribArray(metaballs.positionAttributeLocation);
    gl.vertexAttribPointer(metaballs.positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
};
effects.push(metaballs);

// --- Effect 9: RotoZoomer ---
const rotoZoomer = {};
rotoZoomer.vsSource = quadVS;
rotoZoomer.fsSource = `
    precision highp float;
    uniform vec2 u_resolution;
    uniform float u_time;

    void main() {
        vec2 uv = (gl_FragCoord.xy / u_resolution) * 2.0 - 1.0;
        uv.x *= u_resolution.x / u_resolution.y; // Aspect ratio correction

        float zoom = 1.0 + sin(u_time * 0.5) * 0.5; // Oscillating zoom
        float angle = u_time * 0.5; // Rotation

        // Apply rotation and zoom
        vec2 rotated_uv;
        rotated_uv.x = uv.x * cos(angle) - uv.y * sin(angle);
        rotated_uv.y = uv.x * sin(angle) + uv.y * cos(angle);
        rotated_uv /= zoom;

        // Simple checkerboard pattern
        float check_size = 0.1;
        float c = floor(rotated_uv.x / check_size) + floor(rotated_uv.y / check_size);
        vec3 color = mod(c, 2.0) < 0.5 ? vec3(1.0, 0.0, 1.0) : vec3(0.0, 1.0, 1.0); // Magenta and Cyan

        gl_FragColor = vec4(color, 1.0);
    }
`;

rotoZoomer.init = () => {
    rotoZoomer.program = createProgram(gl,
        createShader(gl, gl.VERTEX_SHADER, rotoZoomer.vsSource),
        createShader(gl, gl.FRAGMENT_SHADER, rotoZoomer.fsSource)
    );
    gl.useProgram(rotoZoomer.program);

    rotoZoomer.positionAttributeLocation = gl.getAttribLocation(rotoZoomer.program, 'a_position');
    rotoZoomer.resolutionUniformLocation = gl.getUniformLocation(rotoZoomer.program, 'u_resolution');
    rotoZoomer.timeUniformLocation = gl.getUniformLocation(rotoZoomer.program, 'u_time');
};

rotoZoomer.draw = (time) => {
    gl.useProgram(rotoZoomer.program);
    gl.uniform2f(rotoZoomer.resolutionUniformLocation, canvas.width, canvas.height);
    gl.uniform1f(rotoZoomer.timeUniformLocation, time / 1000.0);

    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.enableVertexAttribArray(rotoZoomer.positionAttributeLocation);
    gl.vertexAttribPointer(rotoZoomer.positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
};
effects.push(rotoZoomer);

// --- Effect 12: Wave Distortion ---
const waveDistortion = {};
waveDistortion.vsSource = quadVS;
waveDistortion.fsSource = `
    precision highp float;
    uniform vec2 u_resolution;
    uniform float u_time;

    void main() {
        vec2 uv = gl_FragCoord.xy / u_resolution;

        float waveX = sin(uv.y * 20.0 + u_time * 3.0) * 0.05;
        float waveY = cos(uv.x * 20.0 + u_time * 3.0) * 0.05;

        vec2 distortedUv = uv + vec2(waveX, waveY);

        // Simple color based on distorted UV
        vec3 color = vec3(distortedUv.x, distortedUv.y, 0.5);

        gl_FragColor = vec4(color, 1.0);
    }
`;

waveDistortion.init = () => {
    waveDistortion.program = createProgram(gl,
        createShader(gl, gl.VERTEX_SHADER, waveDistortion.vsSource),
        createShader(gl, gl.FRAGMENT_SHADER, waveDistortion.fsSource)
    );
    gl.useProgram(waveDistortion.program);

    waveDistortion.positionAttributeLocation = gl.getAttribLocation(waveDistortion.program, 'a_position');
    waveDistortion.resolutionUniformLocation = gl.getUniformLocation(waveDistortion.program, 'u_resolution');
    waveDistortion.timeUniformLocation = gl.getUniformLocation(waveDistortion.program, 'u_time');
};

waveDistortion.draw = (time) => {
    gl.useProgram(waveDistortion.program);
    gl.uniform2f(waveDistortion.resolutionUniformLocation, canvas.width, canvas.height);
    gl.uniform1f(waveDistortion.timeUniformLocation, time / 1000.0);

    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.enableVertexAttribArray(waveDistortion.positionAttributeLocation);
    gl.vertexAttribPointer(waveDistortion.positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
};
effects.push(waveDistortion);

// --- Effect 8: Fire Flames (Amiga Demoscene Style) ---
const fire = {};
fire.vsSource = quadVS;
fire.fsSource = `
    precision highp float;
    uniform vec2 u_resolution;
    uniform float u_time;

    // Improved noise function for smoother fire
    float noise(vec2 p) {
        vec2 i = floor(p);
        vec2 f = fract(p);
        f = f * f * (3.0 - 2.0 * f);
        float a = fract(sin(dot(i, vec2(12.9898, 78.233))) * 43758.5453);
        float b = fract(sin(dot(i + vec2(1.0, 0.0), vec2(12.9898, 78.233))) * 43758.5453);
        float c = fract(sin(dot(i + vec2(0.0, 1.0), vec2(12.9898, 78.233))) * 43758.5453);
        float d = fract(sin(dot(i + vec2(1.0, 1.0), vec2(12.9898, 78.233))) * 43758.5453);
        return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
    }

    // Fractal Brownian Motion for more natural fire patterns
    float fbm(vec2 p) {
        float value = 0.0;
        float amplitude = 0.5;
        for (int i = 0; i < 4; i++) {
            value += amplitude * noise(p);
            p *= 2.0;
            amplitude *= 0.5;
        }
        return value;
    }

    // Fire palette - more realistic fire colors
    vec3 firePalette(float t) {
        vec3 color = vec3(0.0);
        // Dark red to bright yellow
        color.r = min(1.0, 2.0 * t);
        color.g = min(1.0, 2.0 * t * t);
        color.b = min(1.0, 2.0 * t * t * t);
        return color;
    }

    void main() {
        vec2 uv = gl_FragCoord.xy / u_resolution;
        
        // Flip Y to make fire rise upward
        uv.y = 1.0 - uv.y;
        
        // Time-based movement
        float time = u_time * 0.3;
        
        // Create fire pattern with fbm
        vec2 firePos = vec2(uv.x * 3.0 + time * 0.2, uv.y * 5.0 + time);
        float firePattern = fbm(firePos);
        
        // Shape the fire to be stronger at the bottom
        float fireShape = (1.0 - uv.y) * 1.5;
        firePattern *= fireShape;
        
        // Add some vertical streaks for flame effect
        firePattern += 0.1 * sin(uv.x * 20.0 + time * 2.0) * (1.0 - uv.y);
        
        // Enhance contrast for more defined flames
        firePattern = pow(firePattern, 1.5);
        
        // Apply color palette
        vec3 color = firePalette(firePattern);
        
        // Fade out at the top for more realistic flames
        color *= (1.0 - pow(uv.y, 3.0));
        
        gl_FragColor = vec4(color, 1.0);
    }
`;

fire.init = () => {
    fire.program = createProgram(gl,
        createShader(gl, gl.VERTEX_SHADER, fire.vsSource),
        createShader(gl, gl.FRAGMENT_SHADER, fire.fsSource)
    );
    gl.useProgram(fire.program);

    fire.positionAttributeLocation = gl.getAttribLocation(fire.program, 'a_position');
    fire.resolutionUniformLocation = gl.getUniformLocation(fire.program, 'u_resolution');
    fire.timeUniformLocation = gl.getUniformLocation(fire.program, 'u_time');
};

fire.draw = (time) => {
    gl.useProgram(fire.program);
    gl.uniform2f(fire.resolutionUniformLocation, canvas.width, canvas.height);
    gl.uniform1f(fire.timeUniformLocation, time / 1000.0);

    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.enableVertexAttribArray(fire.positionAttributeLocation);
    gl.vertexAttribPointer(fire.positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
};
effects.push(fire);

// --- WebGL Physics Ball Effect (Converted from Three.js) ---
const physicsBalls = {};
physicsBalls.vsSource = `
    attribute vec3 a_position;
    attribute vec3 a_normal;
    uniform mat4 u_modelViewMatrix;
    uniform mat4 u_projectionMatrix;
    uniform mat3 u_normalMatrix;
    varying vec3 v_normal;
    varying vec3 v_position;
    varying vec3 v_worldPosition;
    
    void main() {
        v_normal = normalize(u_normalMatrix * a_normal);
        vec4 mvPosition = u_modelViewMatrix * vec4(a_position, 1.0);
        v_position = mvPosition.xyz;
        v_worldPosition = a_position;
        gl_Position = u_projectionMatrix * mvPosition;
    }
`;

physicsBalls.fsSource = `
    precision highp float;
    varying vec3 v_normal;
    varying vec3 v_position;
    uniform vec3 u_lightPosition;
    varying vec3 v_worldPosition;
    
    void main() {
        vec3 normal = normalize(v_normal);
        vec3 lightDir = normalize(u_lightPosition - v_position);
        
        // Calculate checkered pattern based on world position
        float patternScale = 2.0;
        vec3 worldPos = v_worldPosition;
        float checkX = mod(floor(worldPos.x * patternScale), 2.0);
        float checkY = mod(floor(worldPos.y * patternScale), 2.0);
        float checkZ = mod(floor(worldPos.z * patternScale), 2.0);
        float checker = mod(checkX + checkY + checkZ, 2.0);
        
        // Red and white checkered pattern
        vec3 baseColor = checker < 0.5 ? vec3(1.0, 0.0, 0.0) : vec3(1.0, 1.0, 1.0);
        
        float diff = max(dot(normal, lightDir), 0.0);
        vec3 diffuse = diff * baseColor;
        
        vec3 ambient = 0.3 * baseColor;
        
        vec3 viewDir = normalize(-v_position);
        vec3 reflectDir = reflect(-lightDir, normal);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
        vec3 specular = spec * vec3(1.0);
        
        vec3 result = ambient + diffuse + specular;
        gl_FragColor = vec4(result, 1.0);
    }
`;

// Matrix math utilities
function createMat4() {
    return new Float32Array(16);
}

function identity(out) {
    out[0] = 1; out[1] = 0; out[2] = 0; out[3] = 0;
    out[4] = 0; out[5] = 1; out[6] = 0; out[7] = 0;
    out[8] = 0; out[9] = 0; out[10] = 1; out[11] = 0;
    out[12] = 0; out[13] = 0; out[14] = 0; out[15] = 1;
    return out;
}

function perspective(out, fovy, aspect, near, far) {
    const f = 1.0 / Math.tan(fovy / 2);
    const nf = 1 / (near - far);
    out[0] = f / aspect; out[1] = 0; out[2] = 0; out[3] = 0;
    out[4] = 0; out[5] = f; out[6] = 0; out[7] = 0;
    out[8] = 0; out[9] = 0; out[10] = (far + near) * nf; out[11] = -1;
    out[12] = 0; out[13] = 0; out[14] = 2 * far * near * nf; out[15] = 0;
    return out;
}

function translate(out, a, v) {
    const x = v[0], y = v[1], z = v[2];
    out[0] = a[0]; out[1] = a[1]; out[2] = a[2]; out[3] = a[3];
    out[4] = a[4]; out[5] = a[5]; out[6] = a[6]; out[7] = a[7];
    out[8] = a[8]; out[9] = a[9]; out[10] = a[10]; out[11] = a[11];
    out[12] = a[0] * x + a[4] * y + a[8] * z + a[12];
    out[13] = a[1] * x + a[5] * y + a[9] * z + a[13];
    out[14] = a[2] * x + a[6] * y + a[10] * z + a[14];
    out[15] = a[3] * x + a[7] * y + a[11] * z + a[15];
    return out;
}

function rotate(out, a, rad, axis) {
    let x = axis[0], y = axis[1], z = axis[2];
    let len = Math.sqrt(x * x + y * y + z * z);
    
    if (Math.abs(len) < 0.000001) { return null; }
    
    len = 1 / len;
    x *= len;
    y *= len;
    z *= len;
    
    const s = Math.sin(rad);
    const c = Math.cos(rad);
    const t = 1 - c;
    
    const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
    const a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
    const a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
    
    const b00 = x * x * t + c, b01 = y * x * t + z * s, b02 = z * x * t - y * s;
    const b10 = x * y * t - z * s, b11 = y * y * t + c, b12 = z * y * t + x * s;
    const b20 = x * z * t + y * s, b21 = y * z * t - x * s, b22 = z * z * t + c;
    
    out[0] = a00 * b00 + a10 * b01 + a20 * b02;
    out[1] = a01 * b00 + a11 * b01 + a21 * b02;
    out[2] = a02 * b00 + a12 * b01 + a22 * b02;
    out[3] = a03 * b00 + a13 * b01 + a23 * b02;
    out[4] = a00 * b10 + a10 * b11 + a20 * b12;
    out[5] = a01 * b10 + a11 * b11 + a21 * b12;
    out[6] = a02 * b10 + a12 * b11 + a22 * b12;
    out[7] = a03 * b10 + a13 * b11 + a23 * b12;
    out[8] = a00 * b20 + a10 * b21 + a20 * b22;
    out[9] = a01 * b20 + a11 * b21 + a21 * b22;
    out[10] = a02 * b20 + a12 * b21 + a22 * b22;
    out[11] = a03 * b20 + a13 * b21 + a23 * b22;
    
    if (a !== out) {
        out[12] = a[12];
        out[13] = a[13];
        out[14] = a[14];
        out[15] = a[15];
    }
    return out;
}

function normalFromMat4(out, a) {
    const a00 = a[0], a01 = a[1], a02 = a[2];
    const a10 = a[4], a11 = a[5], a12 = a[6];
    const a20 = a[8], a21 = a[9], a22 = a[10];
    
    const b00 = a11 * a22 - a12 * a21;
    const b01 = a12 * a20 - a10 * a22;
    const b02 = a10 * a21 - a11 * a20;
    
    let len = Math.sqrt(b00 * b00 + b01 * b01 + b02 * b02);
    if (len) {
        len = 1 / len;
    }
    
    out[0] = b00 * len;
    out[1] = b01 * len;
    out[2] = b02 * len;
    return out;
}

// Create sphere geometry
function createSphere(radius, widthSegments, heightSegments) {
    const positions = [];
    const normals = [];
    const indices = [];
    
    for (let y = 0; y <= heightSegments; y++) {
        const v = y / heightSegments;
        const phi = v * Math.PI;
        
        for (let x = 0; x <= widthSegments; x++) {
            const u = x / widthSegments;
            const theta = u * Math.PI * 2;
            
            const sinPhi = Math.sin(phi);
            const cosPhi = Math.cos(phi);
            const sinTheta = Math.sin(theta);
            const cosTheta = Math.cos(theta);
            
            const nx = cosTheta * sinPhi;
            const ny = cosPhi;
            const nz = sinTheta * sinPhi;
            
            positions.push(radius * nx, radius * ny, radius * nz);
            normals.push(nx, ny, nz);
        }
    }
    
    for (let y = 0; y < heightSegments; y++) {
        for (let x = 0; x < widthSegments; x++) {
            const a = y * (widthSegments + 1) + x;
            const b = a + widthSegments + 1;
            
            indices.push(a, b, a + 1);
            indices.push(b, b + 1, a + 1);
        }
    }
    
    return {
        positions: new Float32Array(positions),
        normals: new Float32Array(normals),
        indices: new Uint16Array(indices)
    };
}

// Vector3 class (simplified)
class Vec3 {
    constructor(x = 0, y = 0, z = 0) {
        this.x = x;
        this.y = y;
        this.z = z;
    }
    
    clone() {
        return new Vec3(this.x, this.y, this.z);
    }
    
    copy(v) {
        this.x = v.x;
        this.y = v.y;
        this.z = v.z;
        return this;
    }
    
    add(v) {
        this.x += v.x;
        this.y += v.y;
        this.z += v.z;
        return this;
    }
    
    sub(v) {
        this.x -= v.x;
        this.y -= v.y;
        this.z -= v.z;
        return this;
    }
    
    multiplyScalar(s) {
        this.x *= s;
        this.y *= s;
        this.z *= s;
        return this;
    }
    
    divideScalar(s) {
        if (s !== 0) {
            this.x /= s;
            this.y /= s;
            this.z /= s;
        }
        return this;
    }
    
    length() {
        return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z);
    }
    
    lengthSq() {
        return this.x * this.x + this.y * this.y + this.z * this.z;
    }
    
    normalize() {
        const l = this.length();
        if (l > 0) {
            this.divideScalar(l);
        }
        return this;
    }
    
    dot(v) {
        return this.x * v.x + this.y * v.y + this.z * v.z;
    }
    
    cross(v) {
        const x = this.y * v.z - this.z * v.y;
        const y = this.z * v.x - this.x * v.z;
        const z = this.x * v.y - this.y * v.x;
        return new Vec3(x, y, z);
    }
    
    negate() {
        return this.multiplyScalar(-1);
    }
}

// Physics simulation variables
let balls = [];
let walls = [];
let camera, renderer, scene;
let sound1, sound2;
let sphereDetail = 32;
let numberOfBalls = 1;
let radius = 2; // Increased ball size
let mouseImpulse = 0.1;
let gravity = new Vec3(0, -0.01, 0);
let controls = {
    Gravity: 0.1,
    Speed: 1
};
controls["e_n Wall"] = 1;
controls["e_t Wall"] = -1;
controls["e_a Wall"] = -1;
controls["e_n Ball"] = 1;
controls["e_t Ball"] = 1;
controls["e_a Ball"] = 1;

// Physics functions
function onWindowResize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    if (camera) {
        camera.aspect = canvas.width / canvas.height;
        camera.updateProjectionMatrix();
    }
}




function init() {
    // Add event listeners
    window.addEventListener("resize", onWindowResize, false);
    
    // Initialize WebGL context
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
    
    // Create camera
    camera = {
        position: new Vec3(0, 0, 22),
        aspect: canvas.width / canvas.height,
        fov: 45,
        near: 0.1,
        far: 1000,
        projectionMatrix: createMat4(),
        viewMatrix: createMat4(),
        updateProjectionMatrix: function() {
            perspective(this.projectionMatrix, 
                this.fov * Math.PI / 180, 
                this.aspect, 
                this.near, 
                this.far
            );
        }
    };
    camera.updateProjectionMatrix();
    
    // Create sphere geometry
    const sphereGeometry = createSphere(radius, sphereDetail, sphereDetail);
    
    // Create ball buffers
    const ballPositionsBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, ballPositionsBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, sphereGeometry.positions, gl.STATIC_DRAW);
    
    const ballNormalsBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, ballNormalsBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, sphereGeometry.normals, gl.STATIC_DRAW);
    
    const ballIndicesBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ballIndicesBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, sphereGeometry.indices, gl.STATIC_DRAW);
    
    // Create ball program
    physicsBalls.program = createProgram(gl,
        createShader(gl, gl.VERTEX_SHADER, physicsBalls.vsSource),
        createShader(gl, gl.FRAGMENT_SHADER, physicsBalls.fsSource)
    );
    gl.useProgram(physicsBalls.program);
    
    // Get attribute and uniform locations
    physicsBalls.positionAttributeLocation = gl.getAttribLocation(physicsBalls.program, 'a_position');
    physicsBalls.normalAttributeLocation = gl.getAttribLocation(physicsBalls.program, 'a_normal');
    physicsBalls.modelViewMatrixLocation = gl.getUniformLocation(physicsBalls.program, 'u_modelViewMatrix');
    physicsBalls.projectionMatrixLocation = gl.getUniformLocation(physicsBalls.program, 'u_projectionMatrix');
    physicsBalls.normalMatrixLocation = gl.getUniformLocation(physicsBalls.program, 'u_normalMatrix');
    physicsBalls.lightPositionLocation = gl.getUniformLocation(physicsBalls.program, 'u_lightPosition');
    physicsBalls.ballColorLocation = gl.getUniformLocation(physicsBalls.program, 'u_ballColor');
    
    // Create single red ball with sideways rotation
    const ball = {
        position: new Vec3(0, 0, 0),
        v: new Vec3(0.03, 0, 0),
        rotation: new Vec3(0, 0, Math.PI / 8),
        w: new Vec3(0, 0.5, 0), // Increased sideways rotation
        r: radius,
        m: 1,
        a: 2 / 3,
        I: 0,
        mp: 0,
        color: [1.0, 0.0, 0.0], // Red
        modelMatrix: createMat4(),
        modelViewMatrix: createMat4(),
        normalMatrix: createMat3()
    };
    
    ball.I = ball.a * ball.m * ball.r * ball.r;
    ball.mp = ball.a * ball.m / (1 + ball.a);
    
    balls.push(ball);
    
    // Create light
    const lightPosition = [0, 21, 0];
    
    // Create ground plane
    const planeGeometry = {
        positions: new Float32Array([
            -11.5, -6.01, -11.5,
             11.5, -6.01, -11.5,
            -11.5, -6.01,  11.5,
             11.5, -6.01,  11.5
        ]),
        indices: new Uint16Array([0, 1, 2, 2, 1, 3])
    };
    
    const planePositionsBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, planePositionsBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, planeGeometry.positions, gl.STATIC_DRAW);
    
    const planeIndicesBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, planeIndicesBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, planeGeometry.indices, gl.STATIC_DRAW);
    
    // Create walls (simplified)
    walls = [];
    for (let i = 0; i < 6; i++) {
        walls.push({
            normal: new Vec3(
                i === 0 ? -1 : i === 1 ? 1 : i === 2 ? 0 : i === 3 ? 0 : i === 4 ? 0 : 0,
                i === 2 ? -1 : i === 3 ? 1 : i === 0 || i === 1 ? 0 : i === 4 ? 0 : 0,
                i === 4 ? -1 : i === 5 ? 1 : 0
            ),
            distance: 6,
            visible: true
        });
    }
    
    
    // Store buffers and geometry references
    physicsBalls.sphereGeometry = sphereGeometry;
    physicsBalls.ballPositionsBuffer = ballPositionsBuffer;
    physicsBalls.ballNormalsBuffer = ballNormalsBuffer;
    physicsBalls.ballIndicesBuffer = ballIndicesBuffer;
    physicsBalls.planePositionsBuffer = planePositionsBuffer;
    physicsBalls.planeIndicesBuffer = planeIndicesBuffer;
    physicsBalls.lightPosition = lightPosition;
}

function updateBall(ball, deltaTime) {
    // Update position
    ball.position.add(
        gravity.clone().multiplyScalar(0.5 * controls.Gravity).add(ball.v).multiplyScalar(deltaTime)
    );
    
    // Update velocity
    ball.v.add(gravity.clone().multiplyScalar(controls.Gravity * deltaTime));
    
    // Update rotation
    const rotationAngle = ball.w.length() * deltaTime;
    if (rotationAngle > 0) {
        const rotationAxis = ball.w.clone().normalize();
        // Apply rotation to ball's orientation (simplified)
    }
}

function updatePhysics() {
    const deltaTime = controls.Speed * 0.016; // Assuming 60 FPS
    
    // Update each ball
    for (let i = 0; i < numberOfBalls; i++) {
        updateBall(balls[i], deltaTime);
    }
    
    // Check wall collisions
    for (let i = 0; i < walls.length; i++) {
        const wall = walls[i];
        for (let j = 0; j < numberOfBalls; j++) {
            const ball = balls[j];
            const distanceToWall = ball.position.dot(wall.normal);
            
            if (distanceToWall > wall.distance - ball.r) {
                // Collision detected
                collideWall(ball, wall);
            }
        }
    }
    
    // Check ball-to-ball collisions
    for (let i = 0; i < numberOfBalls - 1; i++) {
        for (let j = i + 1; j < numberOfBalls; j++) {
            const ball1 = balls[i];
            const ball2 = balls[j];
            const distance = ball1.position.clone().sub(ball2.position).length();
            const minDistance = ball1.r + ball2.r;
            
            if (distance < minDistance) {
                collideBalls(ball1, ball2);
            }
        }
    }
}

function collideWall(ball, wall) {
    const normal = wall.normal;
    const r = normal.clone().multiplyScalar(ball.r);
    const v = ball.v.clone().add(ball.w.clone().cross(r));
    const vn = normal.clone().multiplyScalar(normal.dot(v));
    const vt = v.clone().sub(vn);
    
    const impulse_n = vn.clone().multiplyScalar(-(1 + controls["e_n Wall"]) * ball.m);
    const impulse_t = vt.clone().multiplyScalar((1 + controls["e_t Wall"]) * ball.mp);
    const impulse = impulse_n.clone().add(impulse_t);
    
    const angularImpulse = normal.clone().multiplyScalar(
        -normal.dot(ball.w) * (1 + controls["e_a Wall"]) * ball.I
    );
    
    impulseBall(ball, r, impulse, angularImpulse);
}

function collideBalls(ball1, ball2) {
    const normal = ball2.position.clone().sub(ball1.position).normalize();
    const r1 = normal.clone().multiplyScalar(ball1.r);
    const r2 = normal.clone().multiplyScalar(-ball2.r);
    
    const v1 = ball1.v.clone().add(ball1.w.clone().cross(r1));
    const v2 = ball2.v.clone().add(ball2.w.clone().cross(r2));
    
    const v1n = normal.clone().multiplyScalar(normal.dot(v1));
    const v2n = normal.clone().multiplyScalar(normal.dot(v2));
    
    const m1 = ball1.m;
    const m2 = ball2.m;
    const mp1 = ball1.mp;
    const mp2 = ball2.mp;
    const I1 = ball1.I;
    const I2 = ball2.I;
    
    const c = m1 * m2 / (m1 + m2);
    const d = mp1 * mp2 / (mp1 + mp2);
    const i = I1 * I2 / (I1 + I2);
    
    const deltaVn = v2n.clone().sub(v1n);
    const impulse_n = deltaVn.clone().multiplyScalar((1 + controls["e_n Ball"]) * c);
    
    const deltaVt = v2.clone().sub(v2n).sub(v1.clone().sub(v1n));
    const impulse_t = deltaVt.clone().multiplyScalar((1 + controls["e_t Ball"]) * d);
    
    const impulse = impulse_n.clone().add(impulse_t);
    const angularImpulse = normal.clone().multiplyScalar(
        (normal.dot(ball2.w) - normal.dot(ball1.w)) * (1 + controls["e_a Ball"]) * i
    );
    
    impulseBall(ball1, r1, impulse, angularImpulse);
    impulseBall(ball2, r2, impulse.clone().negate(), angularImpulse.clone().negate());
}

function impulseBall(ball, r, linearImpulse, angularImpulse) {
    ball.v.add(linearImpulse.clone().divideScalar(ball.m));
    ball.w.add(r.clone().cross(linearImpulse).add(angularImpulse).divideScalar(ball.I));
}


function animate(currentTime) {
    updatePhysics();
    
    // Clear the canvas
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.4, 0.4, 0.4, 1.0); // Background color
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    
    // Set up matrices
    const viewMatrix = createMat4();
    identity(viewMatrix);
    
    // Draw ground plane
    gl.useProgram(physicsBalls.program);
    gl.uniformMatrix4fv(physicsBalls.projectionMatrixLocation, false, camera.projectionMatrix);
    gl.uniformMatrix4fv(physicsBalls.modelViewMatrixLocation, false, viewMatrix);
    gl.uniform3fv(physicsBalls.lightPositionLocation, physicsBalls.lightPosition);
    
    gl.bindBuffer(gl.ARRAY_BUFFER, physicsBalls.planePositionsBuffer);
    gl.enableVertexAttribArray(physicsBalls.positionAttributeLocation);
    gl.vertexAttribPointer(physicsBalls.positionAttributeLocation, 3, gl.FLOAT, false, 0, 0);
    
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, physicsBalls.planeIndicesBuffer);
    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
    
    // Draw balls
    for (let i = 0; i < balls.length; i++) {
        const ball = balls[i];
        
        // Create model matrix
        const modelMatrix = createMat4();
        identity(modelMatrix);
        translate(modelMatrix, modelMatrix, [ball.position.x, ball.position.y, ball.position.z]);
        
        // Create model-view matrix
        const modelViewMatrix = createMat4();
        identity(modelViewMatrix);
        translate(modelViewMatrix, modelViewMatrix, [-camera.position.x, -camera.position.y, -camera.position.z]);
        const tempMatrix = createMat4();
        translate(tempMatrix, modelMatrix, [ball.position.x, ball.position.y, ball.position.z]);
        multiplyMatrices(modelViewMatrix, modelViewMatrix, tempMatrix);
        
        // Create normal matrix
        const normalMatrix = createMat3();
        normalFromMat4(normalMatrix, modelViewMatrix);
        
        // Set uniforms
        gl.uniformMatrix4fv(physicsBalls.projectionMatrixLocation, false, camera.projectionMatrix);
        gl.uniformMatrix4fv(physicsBalls.modelViewMatrixLocation, false, modelViewMatrix);
        gl.uniformMatrix3fv(physicsBalls.normalMatrixLocation, false, normalMatrix);
        gl.uniform3fv(physicsBalls.lightPositionLocation, physicsBalls.lightPosition);
        
        // Bind buffers and draw
        gl.bindBuffer(gl.ARRAY_BUFFER, physicsBalls.ballPositionsBuffer);
        gl.enableVertexAttribArray(physicsBalls.positionAttributeLocation);
        gl.vertexAttribPointer(physicsBalls.positionAttributeLocation, 3, gl.FLOAT, false, 0, 0);
        
        gl.bindBuffer(gl.ARRAY_BUFFER, physicsBalls.ballNormalsBuffer);
        gl.enableVertexAttribArray(physicsBalls.normalAttributeLocation);
        gl.vertexAttribPointer(physicsBalls.normalAttributeLocation, 3, gl.FLOAT, false, 0, 0);
        
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, physicsBalls.ballIndicesBuffer);
        gl.drawElements(gl.TRIANGLES, physicsBalls.sphereGeometry.indices.length, gl.UNSIGNED_SHORT, 0);
    }
}

// Matrix multiplication helper
function multiplyMatrices(out, a, b) {
    const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
    const a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
    const a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
    const a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];
    
    let b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
    out[0] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    out[1] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    out[2] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    out[3] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
    
    b0 = b[4]; b1 = b[5]; b2 = b[6]; b3 = b[7];
    out[4] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    out[5] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    out[6] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    out[7] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
    
    b0 = b[8]; b1 = b[9]; b2 = b[10]; b3 = b[11];
    out[8] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    out[9] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    out[10] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    out[11] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
    
    b0 = b[12]; b1 = b[13]; b2 = b[14]; b3 = b[15];
    out[12] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    out[13] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    out[14] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    out[15] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
    return out;
}

// Create mat3 helper
function createMat3() {
    return new Float32Array(9);
}

// Initialize physics balls effect
physicsBalls.init = () => {
    init();
};

physicsBalls.draw = (time) => {
    animate(time);
};

effects.push(physicsBalls);

// --- Main Animation Loop ---
function animateEffects(currentTime) {
    requestAnimationFrame(animateEffects);

    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    const elapsed = currentTime - startTime;
    if (elapsed > effectDuration) {
        startTime = currentTime;
        nextEffect(); // Move to next effect instead of resetting to first
    }

    if (effects.length) {
        const eff = effects[currentEffectIndex];
        if (eff && typeof eff.draw === 'function') {
            eff.draw(currentTime);
        }
    }
}

// --- Simple Mouse Click Handler ---
canvas.style.pointerEvents = 'auto';
canvas.addEventListener('click', (e) => {
    e.preventDefault();
    nextEffect();
});

// Fallback: if canvas is under other elements (z-index -1), allow page clicks to advance effect
document.addEventListener('click', (e) => {
    // Ignore clicks on navigation links to preserve anchor behavior
    const target = e.target;
    if (target && target.closest && target.closest('a')) {
        return;
    }
    // If click did not happen on the effect indicator (which has pointer-events: none, but guard anyway)
    if (effectIndicator && effectIndicator.contains && effectIndicator.contains(target)) {
        return;
    }
    nextEffect();
}, true);

// --- Initialization ---
window.addEventListener('resize', () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    // No need to re-init effects on resize, just update uniforms in draw
});




// Create effect indicator
const effectIndicator = document.createElement('div');
effectIndicator.style.position = 'fixed';
effectIndicator.style.top = '10px';
effectIndicator.style.right = '10px';
effectIndicator.style.background = 'rgba(0, 0, 0, 0.7)';
effectIndicator.style.color = 'white';
effectIndicator.style.padding = '10px';
effectIndicator.style.borderRadius = '5px';
effectIndicator.style.fontSize = '14px';
effectIndicator.style.zIndex = '1000';
effectIndicator.style.pointerEvents = 'none';
document.body.appendChild(effectIndicator);

// Update effect indicator
function updateEffectIndicator() {
    // Derive names defensively from availability/order
    const fallbackNames = ['Starfield','Plasma','Sine Wave','Color Cycle','Tunnel','Metaballs','RotoZoomer','Wave Distortion','Fire','Physics Balls'];
    const names = effects.map((_, i) => fallbackNames[i] ?? `Effect ${i+1}`);
    const name = names[currentEffectIndex] ?? `Effect ${currentEffectIndex + 1}`;
    effectIndicator.textContent = `Effect: ${name} (${currentEffectIndex + 1}/${effects.length || 0})`;
}

// Initial setup for the first effect
if (!gl) {
    // If WebGL is not available, avoid attaching the loop and clicks
    effectIndicator.textContent = 'WebGL not supported';
} else {
    if (effects.length) {
        // Ensure we start with a valid effect
        const eff0 = effects[currentEffectIndex];
        if (eff0 && typeof eff0.init === 'function') {
            eff0.init();
        }
    } else {
        console.warn('No effects were registered.');
    }
    updateEffectIndicator();
    animateEffects(performance.now());
}
