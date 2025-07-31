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

// --- Full-screen Quad (for pixel-based effects like Plasma) ---
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
    currentEffectIndex = (currentEffectIndex + 1) % effects.length;
    effects[currentEffectIndex].init(); // Re-initialize the new effect
    startTime = performance.now(); // Reset timer on manual switch
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

// --- Main Animation Loop ---
function animate(currentTime) {
    requestAnimationFrame(animate);

    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    const elapsed = currentTime - startTime;
    if (elapsed > effectDuration) {
        startTime = currentTime;
        currentEffectIndex = (currentEffectIndex + 1) % effects.length;
        effects[currentEffectIndex].init(); // Re-initialize the new effect
    }

    effects[currentEffectIndex].draw(currentTime);
}

// --- Initialization ---
window.addEventListener('resize', () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    // No need to re-init effects on resize, just update uniforms in draw
});

// Event listeners for manual cycling
window.addEventListener('click', nextEffect);
window.addEventListener('keydown', (event) => {
    if (event.code === 'Space') {
        nextEffect();
    }
});

// Initial setup for the first effect
effects[currentEffectIndex].init();
animate(performance.now());
