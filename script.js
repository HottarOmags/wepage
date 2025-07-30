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

// --- Effect 6: Fire Effect ---
const fire = {};
fire.vsSource = quadVS;
fire.fsSource = `
    precision highp float;
    uniform vec2 u_resolution;
    uniform float u_time;

    // Simple pseudo-random number generator
    float rand(vec2 co) {
        return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
    }

    void main() {
        vec2 uv = gl_FragCoord.xy / u_resolution;
        float y = uv.y;
        float x = uv.x;

        // Base fire noise
        float noise = rand(vec2(x * 10.0, y * 20.0 - u_time * 5.0));
        noise += rand(vec2(x * 20.0, y * 40.0 - u_time * 10.0)) * 0.5;
        noise += rand(vec2(x * 40.0, y * 80.0 - u_time * 20.0)) * 0.25;
        noise = noise / (1.0 + 0.5 + 0.25); // Normalize

        // Apply a vertical gradient to simulate fire rising
        float fire_intensity = pow(y, 2.0) * 2.0; // More intense at the bottom
        fire_intensity *= (1.0 - noise * 0.5); // Add some randomness

        vec3 color = vec3(0.0);
        if (fire_intensity > 0.7) {
            color = mix(vec3(1.0, 0.0, 0.0), vec3(1.0, 1.0, 0.0), (fire_intensity - 0.7) / 0.3); // Red to Yellow
        } else if (fire_intensity > 0.4) {
            color = mix(vec3(0.5, 0.0, 0.0), vec3(1.0, 0.0, 0.0), (fire_intensity - 0.4) / 0.3); // Dark Red to Red
        } else {
            color = mix(vec3(0.0, 0.0, 0.0), vec3(0.5, 0.0, 0.0), fire_intensity / 0.4); // Black to Dark Red
        }

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

// --- Effect 8: Mandelbrot Zoom (re-using high-precision shader) ---
const mandelbrot = {};
mandelbrot.vsSource = quadVS;
mandelbrot.fsSource = `
    precision highp float;

    uniform vec2 u_resolution;
    uniform float u_zoom;
    uniform vec2 u_pan;
    uniform int u_maxIterations;

    // High precision arithmetic functions
    vec2 add(vec2 a, vec2 b) {
        float s = a.x + b.x;
        float e = a.x - s + b.x + a.y + b.y;
        return vec2(s, e);
    }

    vec2 sub(vec2 a, vec2 b) {
        float s = a.x - b.x;
        float e = a.x - s - b.x + a.y - b.y;
        return vec2(s, e);
    }

    vec2 mul(vec2 a, vec2 b) {
        float p = a.x * b.x;
        float e = a.x * b.y + a.y * b.x + a.y * b.y;
        return vec2(p, e);
    }

    vec2 square(vec2 a) {
        float p = a.x * a.x;
        float e = 2.0 * a.x * a.y + a.y * a.y;
        return vec2(p, e);
    }

    vec2 fromFloat(float f) {
        return vec2(f, 0.0);
    }

    vec3 hslToRgb(float h, float s, float l) {
        vec3 rgb;
        if (s == 0.0) {
            rgb = vec3(l); // achromatic
        } else {
            float q = l < 0.5 ? l * (1.0 + s) : l + s - l * s;
            float p = 2.0 * l - q;
            rgb.r = h + 1.0/3.0;
            rgb.g = h;
            rgb.b = h - 1.0/3.0;

            vec3 tempRgb = rgb;
            if (tempRgb.r < 0.0) tempRgb.r += 1.0;
            if (tempRgb.g < 0.0) tempRgb.g += 1.0;
            if (tempRgb.b < 0.0) tempRgb.b += 1.0;
            if (tempRgb.r > 1.0) tempRgb.r -= 1.0;
            if (tempRgb.g > 1.0) tempRgb.g -= 1.0;
            if (tempRgb.b > 1.0) tempRgb.b -= 1.0;

            if (tempRgb.r < 1.0/6.0) rgb.r = p + (q - p) * 6.0 * tempRgb.r;
            else if (tempRgb.r < 1.0/2.0) rgb.r = q;
            else if (tempRgb.r < 2.0/3.0) rgb.r = p + (q - p) * (2.0/3.0 - tempRgb.r) * 6.0;
            else rgb.r = p;

            if (tempRgb.g < 1.0/6.0) rgb.g = p + (q - p) * 6.0 * tempRgb.g;
            else if (tempRgb.g < 1.0/2.0) rgb.g = q;
            else if (tempRgb.g < 2.0/3.0) rgb.g = p + (q - p) * (2.0/3.0 - tempRgb.g) * 6.0;
            else rgb.g = p;

            if (tempRgb.b < 1.0/6.0) rgb.b = p + (q - p) * 6.0 * tempRgb.b;
            else if (tempRgb.b < 1.0/2.0) rgb.b = q;
            else if (tempRgb.b < 2.0/3.0) rgb.b = p + (q - p) * (2.0/3.0 - tempRgb.b) * 6.0;
            else rgb.b = p;
        }
        return rgb;
    }

    void main() {
        vec2 uv = gl_FragCoord.xy / u_resolution; // 0 to 1
        uv = uv * 2.0 - 1.0; // -1 to 1

        // Adjust for aspect ratio
        if (u_resolution.x > u_resolution.y) {
            uv.x *= u_resolution.x / u_resolution.y;
        } else {
            uv.y *= u_resolution.y / u_resolution.x;
        }

        // Apply zoom and pan using high precision
        vec2 scale = fromFloat(2.0 / u_zoom);
        vec2 c_re = add(mul(fromFloat(uv.x), scale), fromFloat(u_pan.x));
        vec2 c_im = add(mul(fromFloat(uv.y), scale), fromFloat(u_pan.y));

        vec2 z_re = fromFloat(0.0);
        vec2 z_im = fromFloat(0.0);

        int n = 0;
        for (int i = 0; i < 256; i++) { // Max iterations in shader loop
            if (i == u_maxIterations) break;
            vec2 z_re2 = square(z_re);
            vec2 z_im2 = square(z_im);

            if (add(z_re2, z_im2).x > 4.0) {
                break;
            }

            z_im = add(mul(fromFloat(2.0), mul(z_re, z_im)), c_im);
            z_re = add(sub(z_re2, z_im2), c_re);
            n++;
        }

        vec3 color;
        if (n == u_maxIterations) {
            color = vec3(0.0, 0.0, 0.0); // Inside Mandelbrot set (black)
        } else {
            // Smooth coloring
            float log_zn = log(add(square(z_re), square(z_im)).x) / 2.0;
            float nu = log(log_zn / log(2.0)) / log(2.0);
            float m = float(n) + 1.0 - nu;

            float hue = mod(m * 0.1, 1.0); // Cycle through hues
            float saturation = 1.0;
            float lightness = 0.5;

            color = hslToRgb(hue, saturation, lightness);
        }

        gl_FragColor = vec4(color, 1.0);
    }
`;

mandelbrot.init = () => {
    mandelbrot.program = createProgram(gl,
        createShader(gl, gl.VERTEX_SHADER, mandelbrot.vsSource),
        createShader(gl, gl.FRAGMENT_SHADER, mandelbrot.fsSource)
    );
    gl.useProgram(mandelbrot.program);

    mandelbrot.positionAttributeLocation = gl.getAttribLocation(mandelbrot.program, 'a_position');
    mandelbrot.resolutionUniformLocation = gl.getUniformLocation(mandelbrot.program, 'u_resolution');
    mandelbrot.zoomUniformLocation = gl.getUniformLocation(mandelbrot.program, 'u_zoom');
    mandelbrot.panUniformLocation = gl.getUniformLocation(mandelbrot.program, 'u_pan');
    mandelbrot.maxIterationsUniformLocation = gl.getUniformLocation(mandelbrot.program, 'u_maxIterations');

    mandelbrot.zoom = 1.0;
    mandelbrot.panX = 0.0;
    mandelbrot.panY = 0.0;
    mandelbrot.maxIterations = 256;
    mandelbrot.zoomSpeed = 1.02;
    mandelbrot.targetZoom = 1e10; // Zoom in for a while
};

mandelbrot.draw = (time) => {
    gl.useProgram(mandelbrot.program);
    gl.uniform2f(mandelbrot.resolutionUniformLocation, canvas.width, canvas.height);
    gl.uniform1f(mandelbrot.zoomUniformLocation, mandelbrot.zoom);
    gl.uniform2f(mandelbrot.panUniformLocation, mandelbrot.panX, mandelbrot.panY);
    gl.uniform1i(mandelbrot.maxIterationsUniformLocation, mandelbrot.maxIterations);

    if (mandelbrot.zoom < mandelbrot.targetZoom) {
        mandelbrot.zoom *= mandelbrot.zoomSpeed;
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.enableVertexAttribArray(mandelbrot.positionAttributeLocation);
    gl.vertexAttribPointer(mandelbrot.positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
};
effects.push(mandelbrot);

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

// --- Effect 10: Dot Tunnel ---
const dotTunnel = {};
dotTunnel.vsSource = quadVS;
dotTunnel.fsSource = `
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

        // Create a grid of dots
        float dot_size = 0.05;
        float grid_x = fract(x * 10.0);
        float grid_y = fract(y * 10.0);

        float dist_to_center = distance(vec2(grid_x, grid_y), vec2(0.5, 0.5));
        float dot_alpha = smoothstep(dot_size, dot_size * 0.5, dist_to_center);

        vec3 color = vec3(dot_alpha); // White dots

        gl_FragColor = vec4(color, 1.0);
    }
`;

dotTunnel.init = () => {
    dotTunnel.program = createProgram(gl,
        createShader(gl, gl.VERTEX_SHADER, dotTunnel.vsSource),
        createShader(gl, gl.FRAGMENT_SHADER, dotTunnel.fsSource)
    );
    gl.useProgram(dotTunnel.program);

    dotTunnel.positionAttributeLocation = gl.getAttribLocation(dotTunnel.program, 'a_position');
    dotTunnel.resolutionUniformLocation = gl.getUniformLocation(dotTunnel.program, 'u_resolution');
    dotTunnel.timeUniformLocation = gl.getUniformLocation(dotTunnel.program, 'u_time');
};

dotTunnel.draw = (time) => {
    gl.useProgram(dotTunnel.program);
    gl.uniform2f(dotTunnel.resolutionUniformLocation, canvas.width, canvas.height);
    gl.uniform1f(dotTunnel.timeUniformLocation, time / 1000.0);

    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.enableVertexAttribArray(dotTunnel.positionAttributeLocation);
    gl.vertexAttribPointer(dotTunnel.positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
};
effects.push(dotTunnel);

// --- Effect 11: Moire Pattern ---
const moire = {};
moire.vsSource = quadVS;
moire.fsSource = `
    precision highp float;
    uniform vec2 u_resolution;
    uniform float u_time;

    void main() {
        vec2 uv = gl_FragCoord.xy / u_resolution;

        float lines1 = sin((uv.x * 50.0 + u_time * 5.0) * 3.14159);
        float lines2 = sin((uv.y * 50.0 + u_time * 5.0) * 3.14159);

        float color = (lines1 + lines2) * 0.5;

        gl_FragColor = vec4(vec3(color), 1.0);
    }
`;

moire.init = () => {
    moire.program = createProgram(gl,
        createShader(gl, gl.VERTEX_SHADER, moire.vsSource),
        createShader(gl, gl.FRAGMENT_SHADER, moire.fsSource)
    );
    gl.useProgram(moire.program);

    moire.positionAttributeLocation = gl.getAttribLocation(moire.program, 'a_position');
    moire.resolutionUniformLocation = gl.getUniformLocation(moire.program, 'u_resolution');
    moire.timeUniformLocation = gl.getUniformLocation(moire.program, 'u_time');
};

moire.draw = (time) => {
    gl.useProgram(moire.program);
    gl.uniform2f(moire.resolutionUniformLocation, canvas.width, canvas.height);
    gl.uniform1f(moire.timeUniformLocation, time / 1000.0);

    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.enableVertexAttribArray(moire.positionAttributeLocation);
    gl.vertexAttribPointer(moire.positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
};
effects.push(moire);

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

// Initial setup for the first effect
effects[currentEffectIndex].init();
animate(performance.now());