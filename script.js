"use strict";

const canvas = document.getElementById('geometric-canvas');
const gl = canvas.getContext('webgl');

if (!gl) {
    console.error('WebGL not supported!');
}

canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

// --- Minimal Matrix Helpers (mat4/mat3) for Boing Ball ---
function createMat4() {
    return [1,0,0,0,
            0,1,0,0,
            0,0,1,0,
            0,0,0,1];
}

function identity(out) {
    out[0]=1; out[1]=0; out[2]=0; out[3]=0;
    out[4]=0; out[5]=1; out[6]=0; out[7]=0;
    out[8]=0; out[9]=0; out[10]=1; out[11]=0;
    out[12]=0; out[13]=0; out[14]=0; out[15]=1;
    return out;
}

function perspective(out, fovy, aspect, near, far) {
    const f = 1.0 / Math.tan(fovy / 2);
    const nf = 1 / (near - far);
    out[0] = f / aspect; out[1]=0; out[2]=0; out[3]=0;
    out[4] = 0; out[5]=f; out[6]=0; out[7]=0;
    out[8] = 0; out[9]=0; out[10]=(far + near)*nf; out[11]=-1;
    out[12]=0; out[13]=0; out[14]=(2*far*near)*nf; out[15]=0;
    return out;
}

function translate(out, a, v) {
    const x=v[0], y=v[1], z=v[2];
    if (out !== a) {
        out[0]=a[0]; out[1]=a[1]; out[2]=a[2]; out[3]=a[3];
        out[4]=a[4]; out[5]=a[5]; out[6]=a[6]; out[7]=a[7];
        out[8]=a[8]; out[9]=a[9]; out[10]=a[10]; out[11]=a[11];
        out[12]=a[12]; out[13]=a[13]; out[14]=a[14]; out[15]=a[15];
    }
    out[12] = a[0]*x + a[4]*y + a[8]*z + a[12];
    out[13] = a[1]*x + a[5]*y + a[9]*z + a[13];
    out[14] = a[2]*x + a[6]*y + a[10]*z + a[14];
    out[15] = a[3]*x + a[7]*y + a[11]*z + a[15];
    return out;
}

function rotate(out, a, rad, axis) {
    let x=axis[0], y=axis[1], z=axis[2];
    let len = Math.hypot(x,y,z);
    if (len < 1e-6) return identity(out);
    len = 1/len; x*=len; y*=len; z*=len;

    const s = Math.sin(rad);
    const c = Math.cos(rad);
    const t = 1 - c;

    const a00=a[0], a01=a[1], a02=a[2], a03=a[3];
    const a10=a[4], a11=a[5], a12=a[6], a13=a[7];
    const a20=a[8], a21=a[9], a22=a[10], a23=a[11];

    const b00 = x*x*t + c,     b01 = y*x*t + z*s, b02 = z*x*t - y*s;
    const b10 = x*y*t - z*s,   b11 = y*y*t + c,   b12 = z*y*t + x*s;
    const b20 = x*z*t + y*s,   b21 = y*z*t - x*s, b22 = z*z*t + c;

    out[0] = a00*b00 + a10*b01 + a20*b02;
    out[1] = a01*b00 + a11*b01 + a21*b02;
    out[2] = a02*b00 + a12*b01 + a22*b02;
    out[3] = a03*b00 + a13*b01 + a23*b02;

    out[4] = a00*b10 + a10*b11 + a20*b12;
    out[5] = a01*b10 + a11*b11 + a21*b12;
    out[6] = a02*b10 + a12*b11 + a22*b12;
    out[7] = a03*b10 + a13*b11 + a23*b12;

    out[8] = a00*b20 + a10*b21 + a20*b22;
    out[9] = a01*b20 + a11*b21 + a21*b22;
    out[10]= a02*b20 + a12*b21 + a22*b22;
    out[11]= a03*b20 + a13*b21 + a23*b22;

    out[12]= a[12];
    out[13]= a[13];
    out[14]= a[14];
    out[15]= a[15];
    return out;
}

function createMat3() {
    return [1,0,0,
            0,1,0,
            0,0,1];
}

function normalFromMat4(out, m) {
    const a00=m[0], a01=m[1], a02=m[2];
    const a10=m[4], a11=m[5], a12=m[6];
    const a20=m[8], a21=m[9], a22=m[10];

    const b01 = a22*a11 - a12*a21;
    const b11 = -a22*a10 + a12*a20;
    const b21 = a21*a10 - a11*a20;

    let det = a00*b01 + a01*b11 + a02*b21;
    if (Math.abs(det) < 1e-8) {
        out[0]=1; out[1]=0; out[2]=0;
        out[3]=0; out[4]=1; out[5]=0;
        out[6]=0; out[7]=0; out[8]=1;
        return out;
    }
    det = 1.0/det;

    out[0] = b01*det;
    out[1] = (-a22*a01 + a02*a21)*det;
    out[2] = (a12*a01 - a02*a11)*det;
    out[3] = b11*det;
    out[4] = (a22*a00 - a02*a20)*det;
    out[5] = (-a12*a00 + a02*a10)*det;
    out[6] = b21*det;
    out[7] = (-a21*a00 + a01*a20)*det;
    out[8] = (a11*a00 - a01*a10)*det;
    return out;
}

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

// Utility: rebuild effects array to match EFFECT_ORDER using effect.name
function applyEffectOrder() {
    if (!effects.length) return;
    const byName = new Map(effects.map(e => [e.name || 'Unnamed', e]));
    const ordered = [];
    for (const name of EFFECT_ORDER) {
        const eff = byName.get(name);
        if (eff) ordered.push(eff);
    }
    // Add any remaining effects not listed in EFFECT_ORDER to the end
    for (const e of effects) {
        if (!ordered.includes(e)) ordered.push(e);
    }
    effects.length = 0;
    for (const e of ordered) effects.push(e);
}

// Utility: rebuild effects array to match EFFECT_ORDER using effect.name
function applyEffectOrder() {
    if (!effects.length) return;
    const byName = new Map(effects.map(e => [e.name || 'Unnamed', e]));
    const ordered = [];
    for (const name of EFFECT_ORDER) {
        const eff = byName.get(name);
        if (eff) ordered.push(eff);
    }
    // Add any remaining effects not listed in EFFECT_ORDER to the end
    for (const e of effects) {
        if (!ordered.includes(e)) ordered.push(e);
    }
    effects.length = 0;
    for (const e of ordered) effects.push(e);
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

// Centralized effect ordering by name. Reorder this array to change play order.
const EFFECT_ORDER = [
    'Starfield',
    'Shadebobs',
    'Plasma',
    'Sine Wave',
    'Color Cycle',
    'Boing Ball',
    'Metaballs',
    'RotoZoomer',
    'Wave Distortion',
    'Fire',
    'Kaleidoscope Tunnel',
    // 'RGB Split Glitch', // removed from rotation
    'Voronoi Flow',
    'Tunnel'
];

// Centralized effect ordering by name. Reorder this array to change play order.
/* removed duplicate EFFECT_ORDER declaration */

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
    console.log('Switched to effect:', currentEffectIndex, eff && eff.name);
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
starfield.name = 'Starfield';
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
plasma.name = 'Plasma';
effects.push(plasma);

// --- Effect 3: Sine Wave ---
const sineWave = {};
sineWave.vsSource = quadVS; // Use the same vertex shader for full-screen quad
sineWave.fsSource = `
    precision highp float;
    uniform vec2 u_resolution;
    uniform float u_time;

    // Smooth minimum (polynomial) for blending multiple distances
    float smin(float a, float b, float k) {
        float h = clamp(0.5 + 0.5*(b - a)/k, 0.0, 1.0);
        return mix(b, a, h) - k*h*(1.0 - h);
    }

    // Color palette helper
    vec3 palette(float t) {
        vec3 a = vec3(0.55, 0.45, 0.65);
        vec3 b = vec3(0.45, 0.55, 0.45);
        vec3 c = vec3(1.00, 1.00, 1.00);
        vec3 d = vec3(0.10, 0.33, 0.67);
        return a + b * cos(6.28318 * (c * (t + d)));
    }

    void main() {
        vec2 uv = gl_FragCoord.xy / u_resolution;

        // Centered space with aspect correction
        vec2 p = uv * 2.0 - 1.0;
        p.x *= u_resolution.x / u_resolution.y;

        float t = u_time;

        // Subtle vertical parallax distortion for depth
        float parallax = 0.04 * sin(p.y * 3.0 + t * 0.7);
        p.x += parallax;

        // Base wave parameters
        float baseFreq = 6.0;
        float speed    = 1.2;
        float amp      = 0.28;

        // Multi-harmonic y target (richer motion)
        float w1 = sin(p.x * baseFreq + t * speed);
        float w2 = 0.5 * sin(p.x * baseFreq * 0.5 - t * 0.9);
        float w3 = 0.33 * sin(p.x * baseFreq * 1.7 + t * 1.7 + 1.2);

        float yTarget = (w1 + w2 + w3) * amp * 0.5;

        // Distance from the wave centerline
        float d = abs(p.y - yTarget);

        // Add a faint secondary wave offset for glow layering
        float yTarget2 = (sin(p.x * (baseFreq * 1.12) + t * (speed * 0.85)) +
                          0.5 * sin(p.x * (baseFreq * 0.56) - t * 1.1)) * amp * 0.35;
        float d2 = abs(p.y - yTarget2);

        // Blend distances for a smooth combined band
        float k = 0.08; // blending softness
        float dist = smin(d, d2, k);

        // Core thickness and soft outer glow
        float core   = 0.008;
        float glow   = 0.12;

        // Smooth alpha for core and glow
        float aCore = 1.0 - smoothstep(core * 0.5, core, dist);
        float aGlow = 1.0 - smoothstep(core, glow, dist);

        // Time-based hue factor influenced by local phase for variation
        float hueShift = 0.25 * sin(t * 0.4) + 0.25 * sin((p.x + p.y) * 1.2 + t * 0.6);
        float phase = 0.5 + 0.5 * sin(p.x * baseFreq + t);
        vec3 colCore = palette(phase + hueShift);
        vec3 colGlow = palette(phase * 0.7 + 0.1 + 0.5 * hueShift);

        // Keep glow a bit cyan-leaning
        colGlow = mix(colGlow, vec3(0.0, 1.0, 1.0), 0.5);

        // Compose on dark background with additive flavor
        vec3 bg = vec3(0.0);
        vec3 color = bg;
        color += colGlow * aGlow * 0.55;
        color += colCore * aCore * 1.0;

        // Subtle scanline shimmer
        float scan = 0.08 * sin(uv.y * u_resolution.y * 3.14159 + t * 3.0);
        color += vec3(scan) * 0.03;

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
sineWave.name = 'Sine Wave';
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
colorCycle.name = 'Color Cycle';
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
tunnel.name = 'Tunnel';
effects.push(tunnel);

// --- Effect 6: Shadebobs (Old-school + Pimped) ---
const shadebobs = {};
shadebobs.vsSource = quadVS;
shadebobs.fsSource = `
    precision highp float;
    uniform vec2 u_resolution;
    uniform float u_time;

    // pseudo-random
    float hash(float n){ return fract(sin(n)*43758.5453123); }
    vec2 hash2(float n){ return fract(sin(vec2(n,n+1.23))*vec2(43758.5453,22578.145912)); }

    // gaussian-ish falloff
    float blob(vec2 p, vec2 c, float r){
        float d = length(p - c);
        float g = exp(- (d*d) / (2.0*r*r));
        return g;
    }

    // neon palette
    vec3 palette(float t){
        vec3 a = vec3(0.55,0.45,0.65);
        vec3 b = vec3(0.45,0.55,0.45);
        vec3 c = vec3(1.00,1.00,1.00);
        vec3 d = vec3(0.10,0.33,0.67);
        return a + b * cos(6.28318 * (c*(t + d)));
    }

    void main(){
        vec2 uv = gl_FragCoord.xy / u_resolution;
        vec2 p = uv * 2.0 - 1.0;
        p.x *= u_resolution.x / u_resolution.y;

        float t = u_time * 0.55;

        // slight domain warp for extra motion
        vec2 warp = vec2(
            0.02 * sin(p.y*6.0 + t*2.0),
            0.02 * cos(p.x*6.0 - t*2.2)
        );
        p += warp;

        // parameters
        const int N = 16;
        float radius = 0.23;
        float accum = 0.0;
        vec3 col = vec3(0.0);

        // additive shadebobs
        for(int i=0;i<N;i++){
            float fi = float(i);
            // Lissajous-ish paths
            float sp1 = 0.6 + 0.3*hash(fi*7.31);
            float sp2 = 0.7 + 0.35*hash(fi*3.77);
            float ph1 = 6.2831853*hash(fi*5.19);
            float ph2 = 6.2831853*hash(fi*9.17);
            float ax  = 0.55 + 0.25*hash(fi*2.01);
            float ay  = 0.40 + 0.35*hash(fi*4.13);

            vec2 c = vec2(
                ax * sin(t*sp1 + ph1),
                ay * sin(t*sp2 + ph2)
            );

            float r = radius * (0.65 + 0.5*hash(fi*1.11));
            float b = blob(p, c, r);
            accum += b;

            float hue = fract(0.15*fi + 0.35*sin(t*0.7 + fi*0.37));
            vec3 bc = palette(hue);
            col += bc * b;
        }

        // normalize/soft threshold for bloom-y look
        float soft = smoothstep(0.12, 0.6, accum);
        col = col / max(1.0, float(N)*0.6);
        col += col * pow(soft, 3.0) * 0.9;

        // subtle chromatic aberration
        vec2 ca = (p)*0.004;
        float r = col.r + 0.10 * texture2DProj(sampler2D(0), vec4(uv+ca,0.0,1.0)).r; // stub: no real sampler, emulate
        float g = col.g;
        float b = col.b + 0.10 * texture2DProj(sampler2D(0), vec4(uv-ca,0.0,1.0)).b;
        col = vec3(r,g,b);

        // vignette
        float vig = 0.92 - 0.55*dot(p,p);
        col *= clamp(vig, 0.25, 1.0);

        // scanlines
        float scan = 0.04 * sin(gl_FragCoord.y*3.14159 + t*3.0);
        col += vec3(scan)*0.03;

        gl_FragColor = vec4(col, 1.0);
    }
`;

// Note: WebGL1 has no default sampler2D bound; above CA "sampler2D" usage is a stub to fake the effect:
// We'll remove the texture fetch but keep a tiny RGB phase offset to emulate CA visually without textures.
shadebobs.fsSource = `
    precision highp float;
    uniform vec2 u_resolution;
    uniform float u_time;

    float hash(float n){ return fract(sin(n)*43758.5453123); }
    vec2 hash2(float n){ return fract(sin(vec2(n,n+1.23))*vec2(43758.5453,22578.145912)); }

    float blob(vec2 p, vec2 c, float r){
        float d = length(p - c);
        return exp(- (d*d) / (2.0*r*r));
    }

    vec3 palette(float t){
        vec3 a = vec3(0.55,0.45,0.65);
        vec3 b = vec3(0.45,0.55,0.45);
        vec3 c = vec3(1.00,1.00,1.00);
        vec3 d = vec3(0.10,0.33,0.67);
        return a + b * cos(6.28318 * (c*(t + d)));
    }

    void main(){
        vec2 uv = gl_FragCoord.xy / u_resolution;
        vec2 p = uv * 2.0 - 1.0;
        p.x *= u_resolution.x / u_resolution.y;

        float t = u_time * 0.55;

        vec2 warp = vec2(
            0.02 * sin(p.y*6.0 + t*2.0),
            0.02 * cos(p.x*6.0 - t*2.2)
        );
        p += warp;

        const int N = 16;
        float radius = 0.23;
        float accum = 0.0;
        vec3 col = vec3(0.0);

        for(int i=0;i<N;i++){
            float fi = float(i);
            float sp1 = 0.6 + 0.3*hash(fi*7.31);
            float sp2 = 0.7 + 0.35*hash(fi*3.77);
            float ph1 = 6.2831853*hash(fi*5.19);
            float ph2 = 6.2831853*hash(fi*9.17);
            float ax  = 0.55 + 0.25*hash(fi*2.01);
            float ay  = 0.40 + 0.35*hash(fi*4.13);

            vec2 c = vec2(
                ax * sin(t*sp1 + ph1),
                ay * sin(t*sp2 + ph2)
            );

            float r = radius * (0.65 + 0.5*hash(fi*1.11));
            float b = blob(p, c, r);
            accum += b;

            float hue = fract(0.15*fi + 0.35*sin(t*0.7 + fi*0.37));
            vec3 bc = palette(hue);
            col += bc * b;
        }

        float soft = smoothstep(0.12, 0.6, accum);
        col = col / max(1.0, float(N)*0.6);
        col += col * pow(soft, 3.0) * 0.9;

        // emulate slight RGB split by phase-shifting color channels with p
        float rOff = 0.03*sin(3.0*p.x + 2.0*p.y + t*1.7);
        float bOff = 0.03*sin(2.7*p.y - 1.4*p.x - t*1.3);
        col.r += rOff * 0.15;
        col.b += bOff * 0.15;

        float vig = 0.92 - 0.55*dot(p,p);
        col *= clamp(vig, 0.25, 1.0);

        float scan = 0.04 * sin(gl_FragCoord.y*3.14159 + t*3.0);
        col += vec3(scan)*0.03;

        gl_FragColor = vec4(max(col, 0.0), 1.0);
    }
`;

shadebobs.init = () => {
    shadebobs.program = createProgram(gl,
        createShader(gl, gl.VERTEX_SHADER, shadebobs.vsSource),
        createShader(gl, gl.FRAGMENT_SHADER, shadebobs.fsSource)
    );
    gl.useProgram(shadebobs.program);
    shadebobs.positionAttributeLocation = gl.getAttribLocation(shadebobs.program, 'a_position');
    shadebobs.resolutionUniformLocation = gl.getUniformLocation(shadebobs.program, 'u_resolution');
    shadebobs.timeUniformLocation = gl.getUniformLocation(shadebobs.program, 'u_time');
};

shadebobs.draw = (time) => {
    gl.useProgram(shadebobs.program);
    gl.uniform2f(shadebobs.resolutionUniformLocation, canvas.width, canvas.height);
    gl.uniform1f(shadebobs.timeUniformLocation, time / 1000.0);

    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.enableVertexAttribArray(shadebobs.positionAttributeLocation);
    gl.vertexAttribPointer(shadebobs.positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
};
shadebobs.name = 'Shadebobs';
effects.push(shadebobs);

// --- Effect 7: Metaballs ---
const metaballs = {};
metaballs.vsSource = quadVS;
metaballs.fsSource = `
    precision highp float;
    uniform vec2 u_resolution;
    uniform float u_time;

    const int NUM_BALLS = 12;
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
        if (value > 6.0) {
            color = vec3(0.0, 0.2, 1.0); // Blue
        } else if (value > 4.0) {
            color = vec3(0.0, 0.4, 1.0); // Lighter Blue
        } else if (value > 3.0) {
            color = vec3(0.0, 0.7, 1.0); // Even Lighter Blue
        } else if (value > 2.0 ) {
            color = vec3(0.0, 1.0, 1.0); // Even Lighter Blue
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
    for (let i = 0; i < 12; i++) {
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
metaballs.name = 'Metaballs';
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
rotoZoomer.name = 'RotoZoomer';
effects.push(rotoZoomer);

// --- Effect 12: Wave Distortion ---
const waveDistortion = {};
waveDistortion.vsSource = quadVS;
waveDistortion.fsSource = `
    precision highp float;
    uniform vec2 u_resolution;
    uniform float u_time;

    // Hash and noise helpers
    float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453); }
    float noise(vec2 p){
        vec2 i = floor(p);
        vec2 f = fract(p);
        f = f*f*(3.0-2.0*f);
        float a = hash(i);
        float b = hash(i + vec2(1.0,0.0));
        float c = hash(i + vec2(0.0,1.0));
        float d = hash(i + vec2(1.0,1.0));
        return mix(mix(a,b,f.x), mix(c,d,f.x), f.y);
    }
    float fbm(vec2 p){
        float v = 0.0;
        float a = 0.5;
        for(int i=0;i<5;i++){
            v += a * noise(p);
            p *= 2.0;
            a *= 0.5;
        }
        return v;
    }

    vec3 palette(float t){
        vec3 a = vec3(0.50, 0.50, 0.55);
        vec3 b = vec3(0.45, 0.35, 0.55);
        vec3 c = vec3(1.00, 1.00, 1.00);
        vec3 d = vec3(0.00, 0.33, 0.67);
        return a + b * cos(6.28318 * (c * (t + d)));
    }

    void main(){
        vec2 uv = gl_FragCoord.xy / u_resolution;
        vec2 p = uv * 2.0 - 1.0;
        p.x *= u_resolution.x / u_resolution.y;

        float t = u_time * 0.7;

        // Layered directional ripples
        float r1 = sin(p.y * 10.0 + t * 2.2);
        float r2 = sin((p.x * 8.0 - p.y * 6.0) + t * 1.3);
        float r3 = cos((p.x * 14.0 + p.y * 12.0) - t * 1.9);
        float ripple = (r1 * 0.55 + r2 * 0.30 + r3 * 0.15);

        // Flow field with fbm for organic distortion
        vec2 flow = vec2(
            fbm(p * 1.2 + vec2(t * 0.25, 0.0)),
            fbm(p * 1.2 + vec2(0.0, t * 0.25))
        );

        // Combine distortions
        float waveX = sin(p.y * 20.0 + t * 3.0 + flow.x * 4.0) * 0.04;
        float waveY = cos(p.x * 22.0 - t * 2.6 + flow.y * 4.0) * 0.04;

        // Add small high-frequency shimmer
        float shimmer = 0.005 * sin((p.x + p.y) * 120.0 + t * 8.0);

        vec2 distorted = p + vec2(waveX, waveY) + ripple * 0.03 + shimmer;

        // Distance bands to form neon contours
        float bands = sin(length(distorted) * 8.0 - t * 2.0);
        float edges = smoothstep(0.2, 0.95, bands * 0.5 + 0.5);

        // Color based on curved domain and time
        float hue = 0.5 + 0.5 * sin(distorted.x * 1.4 + distorted.y * 1.2 + t * 0.6);
        vec3 col = palette(hue);

        // Add neon lines by accentuating edge regions
        float neon = smoothstep(0.75, 0.99, edges);
        vec3 neonCol = mix(vec3(0.0, 0.9, 1.0), vec3(0.8, 0.2, 1.0), hue);
        col = mix(col, neonCol, neon * 0.9);

        // Vignette and subtle scanlines for demoscene vibe
        float vig = 0.9 - 0.6 * length(p);
        col *= clamp(vig, 0.2, 1.0);

        float scan = 0.03 * sin(gl_FragCoord.y * 3.14159 + t * 3.0);
        col += vec3(scan) * 0.04;

        gl_FragColor = vec4(col, 1.0);
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
waveDistortion.name = 'Wave Distortion';
effects.push(waveDistortion);

// --- Effect 13: Kaleidoscope Tunnel ---
const kaleidoscope = {};
kaleidoscope.vsSource = quadVS;
kaleidoscope.fsSource = `
    precision highp float;
    uniform vec2 u_resolution;
    uniform float u_time;

    vec3 palette(float t){
        vec3 a = vec3(0.5,0.5,0.5);
        vec3 b = vec3(0.5,0.5,0.5);
        vec3 c = vec3(1.0,1.0,1.0);
        vec3 d = vec3(0.00,0.33,0.67);
        return a + b * cos(6.28318 * (c * (t + d)));
    }

    void main(){
        vec2 uv = (gl_FragCoord.xy / u_resolution) * 2.0 - 1.0;
        uv.x *= u_resolution.x / u_resolution.y;

        float t = u_time * 0.6;

        // Polar coords
        float r = length(uv);
        float a = atan(uv.y, uv.x);

        // Mirror into kaleidoscope sectors
        float sectors = 8.0;
        float sectorAngle = 6.2831853 / sectors;
        a = mod(a, sectorAngle);
        a = abs(a - sectorAngle * 0.5);

        // Tunnel mapping with swirl
        float z = 1.0 / max(0.001, r);
        float swirl = a + t * 0.8 + 0.35 * sin(z * 0.8 - t * 1.3);

        // Pattern in tunnel space
        float bands = sin(z * 2.2 + swirl * 6.0 - t * 2.0)
                    + 0.5 * sin(z * 4.0 - swirl * 5.0 + t * 1.4);

        float m = bands * 0.5 + 0.5;

        // Colorize
        vec3 col = palette(m);
        // Neon accents on band edges
        float edge = smoothstep(0.75, 0.98, m);
        col += vec3(0.2, 0.8, 1.0) * edge * 0.5;

        // Vignette
        col *= 0.25 + 0.85 * pow(1.0 - clamp(r, 0.0, 1.0), 1.2);

        gl_FragColor = vec4(col,1.0);
    }
`;
kaleidoscope.init = () => {
    kaleidoscope.program = createProgram(gl,
        createShader(gl, gl.VERTEX_SHADER, kaleidoscope.vsSource),
        createShader(gl, gl.FRAGMENT_SHADER, kaleidoscope.fsSource)
    );
    gl.useProgram(kaleidoscope.program);
    kaleidoscope.positionAttributeLocation = gl.getAttribLocation(kaleidoscope.program, 'a_position');
    kaleidoscope.resolutionUniformLocation = gl.getUniformLocation(kaleidoscope.program, 'u_resolution');
    kaleidoscope.timeUniformLocation = gl.getUniformLocation(kaleidoscope.program, 'u_time');
};
kaleidoscope.draw = (time) => {
    gl.useProgram(kaleidoscope.program);
    gl.uniform2f(kaleidoscope.resolutionUniformLocation, canvas.width, canvas.height);
    gl.uniform1f(kaleidoscope.timeUniformLocation, time / 1000.0);
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.enableVertexAttribArray(kaleidoscope.positionAttributeLocation);
    gl.vertexAttribPointer(kaleidoscope.positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
};
kaleidoscope.name = 'Kaleidoscope Tunnel';
effects.push(kaleidoscope);

// --- Effect 14: RGB Split Glitch ---

// --- Effect 15: Voronoi Flow ---
const voronoiFlow = {};
voronoiFlow.vsSource = quadVS;
voronoiFlow.fsSource = `
    precision highp float;
    uniform vec2 u_resolution;
    uniform float u_time;

    // hash helpers
    float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453); }
    vec2  hash2(vec2 p){ return vec2(hash(p), hash(p+13.1)); }

    // animated jittered feature point inside each cell
    vec2 jitter(vec2 cell, float t){
        vec2 h = hash2(cell);
        float ang = 6.2831853 * h.x + t * (0.3 + 0.4*h.y);
        float rad = 0.35 + 0.25 * sin(t*0.7 + h.x*6.0);
        return vec2(cos(ang), sin(ang)) * rad;
    }

    // compute F1 and F2 distances and direction to nearest site
    // returns: F1, F2, dirToNearest
    vec4 voronoiF1F2(vec2 p, float t){
        vec2 g = floor(p);
        vec2 f = fract(p);
        float F1 = 1e9;
        float F2 = 1e9;
        vec2  dir = vec2(0.0);

        for(int j=-1;j<=1;j++){
            for(int i=-1;i<=1;i++){
                vec2 o = vec2(float(i), float(j));
                vec2 site = o + 0.5 + jitter(g + o, t);
                vec2 r = site - f;
                float d = dot(r,r);
                if(d < F1){
                    F2 = F1;
                    F1 = d;
                    dir = r;
                } else if(d < F2){
                    F2 = d;
                }
            }
        }
        return vec4(sqrt(F1), sqrt(F2), dir);
    }

    vec3 palette(float t){
        vec3 a = vec3(0.42,0.40,0.55);
        vec3 b = vec3(0.55,0.45,0.60);
        vec3 c = vec3(1.0,1.0,1.0);
        vec3 d = vec3(0.00,0.33,0.67);
        return a + b * cos(6.28318 * (c*(t + d)));
    }

    void main(){
        vec2 uv = gl_FragCoord.xy / u_resolution;
        vec2 p = uv * 2.0 - 1.0;
        p.x *= u_resolution.x / u_resolution.y;

        float t = u_time * 0.6;

        // gentle domain warp to avoid static grid feel
        vec2 warp = vec2(
            sin(p.y*2.3 + t*0.9),
            cos(p.x*2.7 - t*1.1)
        ) * 0.25;
        vec2 q = (p + warp) * 3.0;

        // compute voronoi
        vec4 vf = voronoiF1F2(q, t);
        float d1 = vf.x;
        float d2 = vf.y;
        vec2  dir = vf.zw;

        // border metric: distance between F1 and F2 gives crisp edges
        float edge = smoothstep(0.02, 0.0, d2 - d1);
        // inner rim near center
        float rim  = 1.0 - smoothstep(0.10, 0.22, d1);

        // advected phase along direction field for flowing color
        float phase = atan(dir.y, dir.x) + length(q) * 0.3 + t*1.4;
        float flow = 0.5 + 0.5 * sin(phase);

        vec3 col = palette(flow);
        col += vec3(0.20, 0.90, 1.00) * edge * 0.85;   // neon borders
        col += vec3(1.00, 0.35, 0.85) * rim  * 0.30;   // soft inner glow

        // softer vignette
        float vig = 0.92 - 0.42 * dot(p,p);
        col *= clamp(vig, 0.35, 1.0);

        gl_FragColor = vec4(col, 1.0);
    }
`;
voronoiFlow.init = () => {
    voronoiFlow.program = createProgram(gl,
        createShader(gl, gl.VERTEX_SHADER, voronoiFlow.vsSource),
        createShader(gl, gl.FRAGMENT_SHADER, voronoiFlow.fsSource)
    );
    gl.useProgram(voronoiFlow.program);
    voronoiFlow.positionAttributeLocation = gl.getAttribLocation(voronoiFlow.program, 'a_position');
    voronoiFlow.resolutionUniformLocation = gl.getUniformLocation(voronoiFlow.program, 'u_resolution');
    voronoiFlow.timeUniformLocation = gl.getUniformLocation(voronoiFlow.program, 'u_time');
};
voronoiFlow.draw = (time) => {
    gl.useProgram(voronoiFlow.program);
    gl.uniform2f(voronoiFlow.resolutionUniformLocation, canvas.width, canvas.height);
    gl.uniform1f(voronoiFlow.timeUniformLocation, time / 1000.0);
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.enableVertexAttribArray(voronoiFlow.positionAttributeLocation);
    gl.vertexAttribPointer(voronoiFlow.positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
};
voronoiFlow.name = 'Voronoi Flow';
effects.push(voronoiFlow);

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
fire.name = 'Fire';
effects.push(fire);

// --- Amiga Boing Ball Effect (WebGL) ---
const boingBall = {};

// Reuse math helpers below if present; otherwise define minimal ones
// We keep the existing math helpers and sphere creation, but we also need UVs.

// Extend sphere generator with UVs for checker calculation in shader
function createSphereWithUV(radius, widthSegments, heightSegments) {
    const positions = [];
    const normals = [];
    const uvs = [];
    const indices = [];

    for (let y = 0; y <= heightSegments; y++) {
        const v = y / heightSegments;
        const phi = v * Math.PI;
        for (let x = 0; x <= widthSegments; x++) {
            const u = x / widthSegments;
            const theta = u * Math.PI * 2.0;

            const sinPhi = Math.sin(phi);
            const cosPhi = Math.cos(phi);
            const sinTheta = Math.sin(theta);
            const cosTheta = Math.cos(theta);

            const nx = cosTheta * sinPhi;
            const ny = cosPhi;
            const nz = sinTheta * sinPhi;

            positions.push(radius * nx, radius * ny, radius * nz);
            normals.push(nx, ny, nz);
            uvs.push(u, v);
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
        uvs: new Float32Array(uvs),
        indices: new Uint16Array(indices)
    };
}

// Vertex shader for boing ball (per-vertex transform, pass varying)
boingBall.vsSource = `
    attribute vec3 a_position;
    attribute vec3 a_normal;
    attribute vec2 a_uv;
    uniform mat4 u_modelViewMatrix;
    uniform mat4 u_projectionMatrix;
    uniform mat3 u_normalMatrix;
    varying vec3 v_normal;
    varying vec3 v_viewPos;
    varying vec2 v_uv;
    void main() {
        vec4 mv = u_modelViewMatrix * vec4(a_position, 1.0);
        v_viewPos = mv.xyz;
        v_normal = normalize(u_normalMatrix * a_normal);
        v_uv = a_uv;
        gl_Position = u_projectionMatrix * mv;
    }
`;

// Fragment shader renders classic red/white checker using UVs; add specular
boingBall.fsSource = `
    precision highp float;
    varying vec3 v_normal;
    varying vec3 v_viewPos;
    varying vec2 v_uv;
    uniform vec3 u_lightPosView; // light in view space
    uniform float u_checkerBands; // number of checks around
    uniform float u_gloss;
    uniform float u_ambient;

    // Classic boing: longitude bands and latitude slices alternating
    vec3 checkerColor(vec2 uv) {
        float u = uv.x; // [0,1]
        float v = uv.y; // [0,1]
        float bandsU = floor(u * u_checkerBands);
        float bandsV = floor(v * (u_checkerBands * 0.5)); // fewer slices vertically
        float c = mod(bandsU + bandsV, 2.0);
        return (c < 0.5) ? vec3(1.0, 0.0, 0.0) : vec3(1.0);
    }

    void main() {
        vec3 N = normalize(v_normal);
        vec3 L = normalize(u_lightPosView - v_viewPos);
        vec3 V = normalize(-v_viewPos);
        vec3 R = reflect(-L, N);

        vec3 base = checkerColor(v_uv);
        float diff = max(dot(N, L), 0.0);
        float spec = pow(max(dot(R, V), 0.0), 64.0) * u_gloss;

        vec3 color = base * (u_ambient + diff) + spec;
        gl_FragColor = vec4(color, 1.0);
    }
`;

// Simple ground + shadow (blob) using another tiny program
boingBall.groundVS = `
    attribute vec3 a_position;
    uniform mat4 u_modelViewMatrix;
    uniform mat4 u_projectionMatrix;
    void main() {
        gl_Position = u_projectionMatrix * (u_modelViewMatrix * vec4(a_position, 1.0));
    }
`;

boingBall.groundFS = `
    precision mediump float;
    uniform vec4 u_color;
    void main() { gl_FragColor = u_color; }
`;

// Mat helpers reused from the existing file
// identity, perspective, translate, rotate, normalFromMat4, multiplyMatrices, createMat3, createMat4 already exist below/above.

boingBall.init = () => {
    // GL state
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);

    // Camera
    boingBall.camera = {
        position: { x: 0, y: 3.5, z: 18 },
        aspect: canvas.width / canvas.height,
        fov: 45,
        near: 0.1,
        far: 1000,
        projectionMatrix: createMat4()
    };
    perspective(
        boingBall.camera.projectionMatrix,
        boingBall.camera.fov * Math.PI / 180,
        boingBall.camera.aspect,
        boingBall.camera.near,
        boingBall.camera.far
    );

    // Geometry
    const sphere = createSphereWithUV(3.0, 48, 32);
    boingBall.sphere = sphere;

    boingBall.vboPos = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, boingBall.vboPos);
    gl.bufferData(gl.ARRAY_BUFFER, sphere.positions, gl.STATIC_DRAW);

    boingBall.vboNrm = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, boingBall.vboNrm);
    gl.bufferData(gl.ARRAY_BUFFER, sphere.normals, gl.STATIC_DRAW);

    boingBall.vboUV = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, boingBall.vboUV);
    gl.bufferData(gl.ARRAY_BUFFER, sphere.uvs, gl.STATIC_DRAW);

    boingBall.ibo = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, boingBall.ibo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, sphere.indices, gl.STATIC_DRAW);

    // Programs
    boingBall.program = createProgram(
        gl,
        createShader(gl, gl.VERTEX_SHADER, boingBall.vsSource),
        createShader(gl, gl.FRAGMENT_SHADER, boingBall.fsSource)
    );
    boingBall.aPos = gl.getAttribLocation(boingBall.program, 'a_position');
    boingBall.aNrm = gl.getAttribLocation(boingBall.program, 'a_normal');
    boingBall.aUV  = gl.getAttribLocation(boingBall.program, 'a_uv');
    boingBall.uMV  = gl.getUniformLocation(boingBall.program, 'u_modelViewMatrix');
    boingBall.uPR  = gl.getUniformLocation(boingBall.program, 'u_projectionMatrix');
    boingBall.uNM  = gl.getUniformLocation(boingBall.program, 'u_normalMatrix');
    boingBall.uLight = gl.getUniformLocation(boingBall.program, 'u_lightPosView');
    boingBall.uBands = gl.getUniformLocation(boingBall.program, 'u_checkerBands');
    boingBall.uGloss = gl.getUniformLocation(boingBall.program, 'u_gloss');
    boingBall.uAmbient = gl.getUniformLocation(boingBall.program, 'u_ambient');

    // Ground
    const g = {
        positions: new Float32Array([
            -20, 0, -20,
             20, 0, -20,
            -20, 0,  20,
             20, 0,  20
        ]),
        indices: new Uint16Array([0,1,2,2,1,3])
    };
    boingBall.gVBO = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, boingBall.gVBO);
    gl.bufferData(gl.ARRAY_BUFFER, g.positions, gl.STATIC_DRAW);
    boingBall.gIBO = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, boingBall.gIBO);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, g.indices, gl.STATIC_DRAW);

    boingBall.gProgram = createProgram(
        gl,
        createShader(gl, gl.VERTEX_SHADER, boingBall.groundVS),
        createShader(gl, gl.FRAGMENT_SHADER, boingBall.groundFS)
    );
    boingBall.gAPos = gl.getAttribLocation(boingBall.gProgram, 'a_position');
    boingBall.gUMV  = gl.getUniformLocation(boingBall.gProgram, 'u_modelViewMatrix');
    boingBall.gUPR  = gl.getUniformLocation(boingBall.gProgram, 'u_projectionMatrix');
    boingBall.gUCol = gl.getUniformLocation(boingBall.gProgram, 'u_color');

    // Shadow: draw a simple flattened disc under the ball using same ground program but different geometry
    const disc = (() => {
        const segments = 48;
        const r = 3.0;
        const verts = [0,0,0];
        const idx = [];
        for (let i=0;i<=segments;i++){
            const a = (i/segments)*Math.PI*2;
            verts.push(Math.cos(a)*r, 0, Math.sin(a)*r);
        }
        for (let i=1;i<segments;i++){
            idx.push(0, i, i+1);
        }
        // close
        idx.push(0, segments, 1);
        return {
            positions: new Float32Array(verts),
            indices: new Uint16Array(idx)
        };
    })();
    boingBall.sVBO = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, boingBall.sVBO);
    gl.bufferData(gl.ARRAY_BUFFER, disc.positions, gl.STATIC_DRAW);
    boingBall.sIBO = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, boingBall.sIBO);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, disc.indices, gl.STATIC_DRAW);
    boingBall.sCount = disc.indices.length;

    // Animation params
    boingBall.start = performance.now();
    boingBall.rotSpeed = 1.8;   // radians/sec
    boingBall.bounceH = 4.0;    // height amplitude
    boingBall.bounceBase = 3.0; // base height
    boingBall.horzAmp = 3.0;    // subtle horizontal drift
    boingBall.lightView = new Float32Array([10, 12, 10]); // simple fixed light in view space
};

boingBall.draw = (tms) => {
    const t = (tms - boingBall.start) / 1000.0;
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.05, 0.05, 0.08, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // Camera view matrix (look-at simplified with translate)
    const view = createMat4();
    identity(view);
    translate(view, view, [-boingBall.camera.position.x, -boingBall.camera.position.y, -boingBall.camera.position.z]);

    // Ground draw
    gl.useProgram(boingBall.gProgram);
    gl.uniformMatrix4fv(boingBall.gUPR, false, boingBall.camera.projectionMatrix);
    gl.uniformMatrix4fv(boingBall.gUMV, false, view);
    gl.uniform4f(boingBall.gUCol, 0.10, 0.10, 0.12, 1.0);
    gl.bindBuffer(gl.ARRAY_BUFFER, boingBall.gVBO);
    gl.enableVertexAttribArray(boingBall.gAPos);
    gl.vertexAttribPointer(boingBall.gAPos, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, boingBall.gIBO);
    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);

    // Ball transform
    const mv = createMat4();
    identity(mv);
    // view first
    for (let i=0;i<16;i++) mv[i] = view[i];
    // position
    const y = boingBall.bounceBase + Math.abs(Math.sin(t * 2.4)) * boingBall.bounceH;
    const x = Math.sin(t * 0.9) * boingBall.horzAmp;
    translate(mv, mv, [x, y, 0]);
    // rotations
    rotate(mv, mv, t * boingBall.rotSpeed, [0,1,0]);
    rotate(mv, mv, t * boingBall.rotSpeed * 0.7, [1,0,0]);

    // Normal matrix
    const nrm = createMat3();
    normalFromMat4(nrm, mv);

    // Draw shadow disc (flattened, semi-transparent)
    gl.useProgram(boingBall.gProgram);
    const shadowMV = createMat4();
    identity(shadowMV);
    for (let i=0;i<16;i++) shadowMV[i] = view[i];
    translate(shadowMV, shadowMV, [x, 0.01, 0]); // tiny lift to avoid z-fight
    // Scale-like flatten: approximate by non-uniform via manual columns (simple hack: compress y in shader not available; keep disc thin)
    gl.uniformMatrix4fv(boingBall.gUPR, false, boingBall.camera.projectionMatrix);
    gl.uniformMatrix4fv(boingBall.gUMV, false, shadowMV);
    // opacity based on height for softness
    const alpha = Math.max(0.15, 0.55 - y * 0.03);
    gl.uniform4f(boingBall.gUCol, 0.0, 0.0, 0.0, alpha);
    gl.bindBuffer(gl.ARRAY_BUFFER, boingBall.sVBO);
    gl.enableVertexAttribArray(boingBall.gAPos);
    gl.vertexAttribPointer(boingBall.gAPos, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, boingBall.sIBO);
    gl.disable(gl.CULL_FACE); // ensure disc visible
    gl.drawElements(gl.TRIANGLES, boingBall.sCount, gl.UNSIGNED_SHORT, 0);
    gl.enable(gl.CULL_FACE);

    // Draw ball
    gl.useProgram(boingBall.program);
    gl.uniformMatrix4fv(boingBall.uPR, false, boingBall.camera.projectionMatrix);
    gl.uniformMatrix4fv(boingBall.uMV, false, mv);
    gl.uniformMatrix3fv(boingBall.uNM, false, nrm);
    gl.uniform3fv(boingBall.uLight, boingBall.lightView);
    gl.uniform1f(boingBall.uBands, 12.0);
    gl.uniform1f(boingBall.uGloss, 0.7);
    gl.uniform1f(boingBall.uAmbient, 0.25);

    gl.bindBuffer(gl.ARRAY_BUFFER, boingBall.vboPos);
    gl.enableVertexAttribArray(boingBall.aPos);
    gl.vertexAttribPointer(boingBall.aPos, 3, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, boingBall.vboNrm);
    gl.enableVertexAttribArray(boingBall.aNrm);
    gl.vertexAttribPointer(boingBall.aNrm, 3, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, boingBall.vboUV);
    gl.enableVertexAttribArray(boingBall.aUV);
    gl.vertexAttribPointer(boingBall.aUV, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, boingBall.ibo);
    gl.drawElements(gl.TRIANGLES, boingBall.sphere.indices.length, gl.UNSIGNED_SHORT, 0);
};

boingBall.name = 'Boing Ball';
effects.push(boingBall);

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
    const name = (effects[currentEffectIndex] && effects[currentEffectIndex].name) ? effects[currentEffectIndex].name : `Effect ${currentEffectIndex + 1}`;
    effectIndicator.textContent = `Effect: ${name} (${currentEffectIndex + 1}/${effects.length || 0})`;
}

// Initial setup for the first effect
if (!gl) {
    // If WebGL is not available, avoid attaching the loop and clicks
    effectIndicator.textContent = 'WebGL not supported';
} else {
    // Apply centralized ordering before starting
    applyEffectOrder();
    // Start from beginning of the ordered list
    currentEffectIndex = 0;

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
