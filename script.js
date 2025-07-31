"use strict";

const canvas = document.getElementById('geometric-canvas');
const gl = canvas.getContext('webgl');

if (!gl) {
    console.error('WebGL not supported!');
}

canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

// Debug toggle and FPS tracking
const DEBUG_INFO = true; // set to true/false to show/hide on-screen effect info
let fps = 0, fpsAccumTime = 0, fpsFrames = 0, lastFrameTime = performance.now();


const EFFECT_ORDER = [
    'Chromatic Voronoi Bloom',
    'Sine Wave',
    'Starfield',
    'Hyperspace Glyphs',
    'Plasma',
    'Boing Ball',
    'Water',
    'Tunnel',
    'Shadebobs',
    'Neon Fractal Bloom',
    'Flame 2025',
    'Color Cycle',
    'Neon Smoke Portal',
    'Metaballs',
    'RotoZoomer',
    'Wave Distortion',
    'Kaleidoscope Tunnel',
    'Ember Flock',
    'Neon Parallax City',
    'Voronoi Flow',
    'Particle Vector Field',
    'Hex Lattice Warp'
];

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

// (removed duplicate applyEffectOrder)

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
const effectDuration = 7500; // in milliseconds

// EFFECT_ORDER moved to top; keep only one declaration.

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

// --- Effect 0: Neon Fractal Bloom (new demoscene raymarch) ---
const neonFractalBloom = {};
neonFractalBloom.vsSource = quadVS;
neonFractalBloom.fsSource = `
    precision highp float;
    uniform vec2 u_resolution;
    uniform float u_time;

    // Utility
    mat2 rot2(float a){ return mat2(cos(a), -sin(a), sin(a), cos(a)); }

    // Hash/noise for subtle grain
    float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7)))*43758.5453); }

    // Palette distinct from others: cobalt -> neon green -> cyan -> pink
    vec3 palette(float t){
        vec3 c1 = vec3(0.10, 0.20, 0.95);
        vec3 c2 = vec3(0.10, 0.95, 0.30);
        vec3 c3 = vec3(0.05, 0.95, 0.95);
        vec3 c4 = vec3(0.95, 0.25, 0.75);
        if (t < 0.33){
            float k = smoothstep(0.0, 0.33, t);
            return mix(c1, c2, k);
        } else if (t < 0.66){
            float k = smoothstep(0.33, 0.66, t);
            return mix(c2, c3, k);
        } else {
            float k = smoothstep(0.66, 1.0, t);
            return mix(c3, c4, k);
        }
    }

    // Distance estimator: Mandelbulb-esque
    float deMandel(vec3 p){
        vec3 z = p;
        float dr = 1.0;
        float r = 0.0;
        const int ITER = 10;
        for(int i=0;i<ITER;i++){
            r = length(z);
            if (r > 2.5) break;
            // Spherical coords
            float theta = acos(clamp(z.z/r, -1.0, 1.0));
            float phi   = atan(z.y, z.x);
            float power = 8.0;
            float r7 = pow(r, power-1.0);
            dr = r7*power*dr + 1.0;

            // scale and rotate
            theta *= power;
            phi   *= power;

            float st = sin(theta), ct = cos(theta);
            float sp = sin(phi),   cp = cos(phi);
            z = r7*r * vec3(cp*st, sp*st, ct);

            // domain warp to make it more smokey and unique
            z += 0.35*sin(vec3(
                0.9*z.x + 0.6*z.y + 0.5*z.z,
                0.6*z.x - 0.7*z.y + 0.8*z.z,
                -0.4*z.x + 0.8*z.y + 0.7*z.z
            ));
            z += p;
        }
        return 0.5*log(r)*r/dr;
    }

    // Normal from DE
    vec3 getNormal(vec3 p){
        float e = 0.0015;
        vec2 h = vec2(1.0, -1.0) * 0.5773;
        return normalize(
            h.xyy*deMandel(p + h.xyy*e) +
            h.yyx*deMandel(p + h.yyx*e) +
            h.yxy*deMandel(p + h.yxy*e) +
            h.xxx*deMandel(p + h.xxx*e)
        );
    }

    // Soft AO by sampling de
    float softAO(vec3 p, vec3 n){
        float ao = 0.0;
        float sc = 0.008;
        for (int i=1;i<=5;i++){
            float d = deMandel(p + n * (float(i)*sc));
            ao += float(i) * d;
        }
        ao = clamp(ao*0.6, 0.0, 1.0);
        return ao;
    }

    // Raymarch
    vec4 raymarch(vec3 ro, vec3 rd, float t){
        float total = 0.0;
        float glow = 0.0;
        vec3 accum = vec3(0.0);

        // Temporal jitter to reduce banding
        float seed = hash(gl_FragCoord.xy + t);
        total += seed * 0.02;

        for (int i=0;i<120;i++){
            vec3 pos = ro + rd * total;
            // minor camera-centric swirl for demoscene motion
            float ang = 0.15*sin(t*0.37) + 0.09*sin(t*0.21);
            pos.xz = rot2(ang) * pos.xz;

            float d = deMandel(pos);
            float adv = clamp(d, 0.002, 0.25);
            total += adv;

            // accumulate neon-like glow near surface
            float edge = exp(-12.0*abs(d));
            float hue = 0.5 + 0.5*sin(0.6*pos.x + 0.7*pos.y + 0.5*pos.z + t*0.5);
            vec3 c = palette(hue);
            accum += c * edge * 0.04;
            glow += edge * 0.03;

            if (d < 0.0015 || total > 12.0) break;
        }

        vec3 col = accum;
        if (total <= 12.0){
            vec3 p = ro + rd * total;
            vec3 n = getNormal(p);
            vec3 L = normalize(vec3(0.7, 0.9, 0.4));
            float diff = max(dot(n, L), 0.0);
            float spec = pow(max(dot(reflect(-L, n), -rd), 0.0), 48.0);

            float hue = 0.5 + 0.5*sin(0.9*p.x + 0.7*p.y + 0.6*p.z);
            vec3 base = palette(hue);
            float ao = softAO(p, n);

            col = base*(0.18 + 1.1*diff)*ao + spec*vec3(1.0);
            col += vec3(glow)*0.6;
        }

        // Background blend
        vec3 bg = vec3(0.01, 0.015, 0.03);
        col = mix(bg, col, clamp(glow*1.6, 0.0, 1.0));

        // Vignette and scanlines
        // reconstruct NDC p for vignette
        // this matches main() computation
        return vec4(col, 1.0);
    }

    void main(){
        vec2 R = u_resolution;
        vec2 uv = (gl_FragCoord.xy / R)*2.0 - 1.0;
        uv.x *= R.x / R.y;

        float t = u_time * 0.8;

        // Camera
        float r = 3.6;
        float a = 0.45*t;
        vec3 ro = vec3(r*cos(a), 1.2 + 0.3*sin(t*0.4), r*sin(a));
        vec3 ta = vec3(0.0, 0.0, 0.0);

        vec3 ww = normalize(ta - ro);
        vec3 uu = normalize(cross(vec3(0.0,1.0,0.0), ww));
        vec3 vv = cross(ww, uu);

        float fov = 1.3;
        vec3 rd = normalize(uu*uv.x + vv*uv.y + ww*fov);

        vec4 col = raymarch(ro, rd, t);

        // final post: vignette + scan + grain
        float vig = 0.92 - 0.55*dot(uv, uv);
        col.rgb *= clamp(vig, 0.25, 1.0);

        float scan = 0.012 * sin(gl_FragCoord.y * 3.14159 + t * 3.0);
        col.rgb += vec3(scan) * 0.010;

        float g = hash(gl_FragCoord.xy + t) - 0.5;
        col.rgb += g * 0.006;

        gl_FragColor = vec4(max(col.rgb, 0.0), 1.0);
    }
`;
neonFractalBloom.init = () => {
    neonFractalBloom.program = createProgram(gl,
        createShader(gl, gl.VERTEX_SHADER, neonFractalBloom.vsSource),
        createShader(gl, gl.FRAGMENT_SHADER, neonFractalBloom.fsSource)
    );
    gl.useProgram(neonFractalBloom.program);
    neonFractalBloom.positionAttributeLocation = gl.getAttribLocation(neonFractalBloom.program, 'a_position');
    neonFractalBloom.resolutionUniformLocation = gl.getUniformLocation(neonFractalBloom.program, 'u_resolution');
    neonFractalBloom.timeUniformLocation = gl.getUniformLocation(neonFractalBloom.program, 'u_time');
};
neonFractalBloom.draw = (time) => {
    gl.useProgram(neonFractalBloom.program);
    gl.uniform2f(neonFractalBloom.resolutionUniformLocation, canvas.width, canvas.height);
    gl.uniform1f(neonFractalBloom.timeUniformLocation, time / 1000.0);
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.enableVertexAttribArray(neonFractalBloom.positionAttributeLocation);
    gl.vertexAttribPointer(neonFractalBloom.positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
};
neonFractalBloom.name = 'Neon Fractal Bloom';
effects.push(neonFractalBloom);

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

    // Gentle value noise + fbm (2-3 octaves for softness)
    float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453123); }
    float noise(vec2 p){
        vec2 i = floor(p);
        vec2 f = fract(p);
        f = f*f*(3.0-2.0*f);
        float a = hash(i);
        float b = hash(i+vec2(1.0,0.0));
        float c = hash(i+vec2(0.0,1.0));
        float d = hash(i+vec2(1.0,1.0));
        return mix(mix(a,b,f.x), mix(c,d,f.x), f.y);
    }
    float fbm(vec2 p){
        float v = 0.0;
        float a = 0.5;
        mat2 m = mat2(1.6,1.2,-1.2,1.6);
        for(int i=0;i<3;i++){
            v += a * noise(p);
            p = m * p;
            a *= 0.5;
        }
        return v;
    }

    // Green-centered dreamy palette: deep teal -> jade -> mint -> pale lime
    vec3 palette(float t){
        t = clamp(t, 0.0, 1.0);
        vec3 c1 = vec3(0.04, 0.30, 0.28);
        vec3 c2 = vec3(0.08, 0.60, 0.40);
        vec3 c3 = vec3(0.55, 0.95, 0.80);
        vec3 c4 = vec3(0.85, 1.00, 0.70);
        vec3 a = mix(c1, c2, smoothstep(0.00, 0.40, t));
        vec3 b = mix(c2, c3, smoothstep(0.30, 0.80, t));
        vec3 c = mix(c3, c4, smoothstep(0.70, 1.00, t));
        return mix(mix(a, b, 0.6), c, smoothstep(0.65, 1.0, t));
    }

    float softCircle(vec2 p, float r){
        float d = length(p);
        return exp(- (d*d) / (r*r));
    }

    mat2 rot(float a){ return mat2(cos(a), -sin(a), sin(a), cos(a)); }

    void main(){
        vec2 R = u_resolution;
        vec2 uv = gl_FragCoord.xy / R;

        // Centered aspect-corrected coords
        vec2 p = uv * 2.0 - 1.0;
        p.x *= R.x / R.y;

        float t = u_time;

        // Stronger sense of motion: add slow global rotation
        float ang = 0.10*sin(t*0.30) + 0.07*sin(t*0.17);
        p = rot(ang) * p;

        // Domain warps (slightly stronger)
        vec2 w1 = vec2(
            fbm(p*0.9 + vec2( 0.14*t, -0.10*t)),
            fbm(p*0.9 + vec2(-0.11*t,  0.12*t))
        ) - 0.5;
        vec2 w2 = vec2(
            fbm(p*1.7 + vec2( 0.22*t,  0.18*t)),
            fbm(p*1.7 + vec2(-0.19*t, -0.16*t))
        ) - 0.5;

        vec2 q = p + 0.32*w1 + 0.16*w2;
        q.y += 0.10*sin(t*0.36) + 0.07*sin(q.x*1.7 - t*0.42);

        // Ribbons: increase contrast and amplitude for more activity
        float rib = 0.5 + 0.5*sin(q.y*2.6 + 1.0*sin(q.x*1.05) - t*0.55);
        rib = smoothstep(0.20, 0.80, rib);
        float ribEdge = smoothstep(0.70, 0.98, rib); // soft contour accent

        // Tone in green family with a bit more variation
        float tone = 0.55
                   + 0.26*fbm(q*0.95 + vec2(0.16*t, -0.12*t))
                   + 0.20*sin(0.60*q.x - 0.42*q.y + 0.34*t);
        tone = clamp(tone, 0.0, 1.0);

        vec3 base = palette(tone);

        // Layered glows
        float g0 = softCircle(q*vec2(0.9,1.1), 1.40);
        float g1 = softCircle(q + 0.22*vec2(sin(t*0.21), cos(t*0.17)), 1.05);
        float g2 = softCircle(q - 0.20*vec2(cos(t*0.16), sin(t*0.19)), 0.86);

        vec3 tint1 = palette(clamp(tone*0.80 + 0.12, 0.0, 1.0));
        vec3 tint2 = palette(clamp(tone*0.60 + 0.28, 0.0, 1.0));

        vec3 col = base * (0.50 + 0.65*rib);       // stronger ribbon influence
        col += tint1 * g1 * 0.42;
        col += tint2 * g2 * 0.34;
        col = mix(col, col * (0.80 + 0.20*g0), 0.58);

        // Subtle contour bands to increase structure without harsh edges
        float contour = 0.5 + 0.5*sin(length(q)*6.0 - t*0.8);
        contour = smoothstep(0.40, 0.85, contour);
        col = mix(col, col * (0.85 + 0.15*contour), 0.5);

        // Keep center softened
        float center = exp(-dot(p, p) / 0.22);
        float centerSoft = exp(-dot(p, p) / 0.55);
        float centerDamp = mix(1.0, 0.78, center);
        centerDamp = mix(centerDamp, 0.88, centerSoft*0.6);
        col *= centerDamp;

        // Gentle vertical blur mix for dreaminess (reduced to keep activity visible)
        vec2 px = 1.0 / R;
        vec3 nb =
              palette(clamp(tone + 0.012*fbm((q+vec2(0.0, 2.0*px.y))*1.0), 0.0, 1.0)) * 0.22
            + palette(clamp(tone + 0.008*fbm((q+vec2(0.0, 1.0*px.y))*1.1), 0.0, 1.0)) * 0.30
            + col * 0.48;
        col = mix(col, nb, 0.22);

        // Vignette
        float vig = 0.97 - 0.42*dot(p, p);
        col *= clamp(vig, 0.40, 1.0);

        // Very faint film haze
        float haze = (hash(gl_FragCoord.xy + t) - 0.5) * 0.008;
        col += vec3(haze) * 0.030;

        gl_FragColor = vec4(max(col, 0.0), 1.0);
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

// --- Effect: Water (Modern waves + ripples) ---
const water = {};
water.vsSource = quadVS;
water.fsSource = `
   precision highp float;
   uniform vec2 u_resolution;
   uniform float u_time;

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
       float v=0.0, a=0.5;
       mat2 m = mat2(1.6,1.2,-1.2,1.6);
       for(int i=0;i<4;i++){
           v += a*noise(p);
           p = m*p;
           a *= 0.5;
       }
       return v;
   }

   float wave(vec2 p, vec2 dir, float freq, float amp, float speed, float t){
       float phase = dot(p, normalize(dir)) * freq + t*speed;
       return amp * sin(phase);
   }

   float heightAt(vec2 p, float t){
       float h = 0.0;
       h += wave(p, vec2(1.0, 0.2), 5.5, 0.06, 1.2, t);
       h += wave(p, vec2(-0.7, 0.9), 7.0, 0.045, 1.6, t);
       h += wave(p, vec2(0.3, -1.0), 9.5, 0.03, 2.0, t);
       h += wave(p, vec2(1.0, 1.0), 13.0, 0.015, 2.7, t);
       float r = length(p);
       h += 0.012 * sin(60.0*r - t*6.0);
       float sh = fbm(p*3.0 + vec2(t*0.15, -t*0.12));
       h += 0.01*(sh - 0.5);
       return h;
   }

   void main(){
       vec2 uv = gl_FragCoord.xy / u_resolution;
       vec2 p = uv*2.0 - 1.0;
       p.x *= u_resolution.x / u_resolution.y;

       float t = u_time;

       float h = heightAt(p, t);

       float e = 0.002;
       float hx = (heightAt(p + vec2(e,0.0), t) - h)/e;
       float hy = (heightAt(p + vec2(0.0,e), t) - h)/e;

       vec3 N = normalize(vec3(-hx, -hy, 1.0));
       vec3 L = normalize(vec3(0.2, 0.4, 0.9));
       vec3 V = normalize(vec3(0.0, 0.0, 1.0));
       vec3 H = normalize(L + V);

       float diff = max(dot(N, L), 0.0);
       float spec = pow(max(dot(N, H), 0.0), 64.0) * 0.6;

       vec3 deep = vec3(0.02, 0.22, 0.36);
       vec3 shallow = vec3(0.05, 0.55, 0.75);
       float depthMix = clamp(0.5 + h*6.0, 0.0, 1.0);
       vec3 baseCol = mix(deep, shallow, depthMix);

       float bands = 0.5 + 0.5*sin(10.0*h - t*2.0);
       vec3 caustic = mix(vec3(0.0), vec3(0.1, 0.5, 0.9), smoothstep(0.6, 0.95, bands));

       vec3 color = baseCol * (0.25 + 0.85*diff) + spec + caustic*0.6;

       float vig = 0.96 - 0.35*dot(p,p);
       color *= clamp(vig, 0.3, 1.0);
       float scan = 0.02 * sin(gl_FragCoord.y*3.14159 + t*3.0);
       color += vec3(scan)*0.02;

       gl_FragColor = vec4(max(color, 0.0), 1.0);
   }
`;
water.init = () => {
   water.program = createProgram(gl,
       createShader(gl, gl.VERTEX_SHADER, water.vsSource),
       createShader(gl, gl.FRAGMENT_SHADER, water.fsSource)
   );
   gl.useProgram(water.program);
   water.positionAttributeLocation = gl.getAttribLocation(water.program, 'a_position');
   water.resolutionUniformLocation = gl.getUniformLocation(water.program, 'u_resolution');
   water.timeUniformLocation = gl.getUniformLocation(water.program, 'u_time');
};
water.draw = (time) => {
   gl.useProgram(water.program);
   gl.uniform2f(water.resolutionUniformLocation, canvas.width, canvas.height);
   gl.uniform1f(water.timeUniformLocation, time / 1000.0);
   gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
   gl.enableVertexAttribArray(water.positionAttributeLocation);
   gl.vertexAttribPointer(water.positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);
   gl.drawArrays(gl.TRIANGLES, 0, 6);
};
water.name = 'Water';
effects.push(water);

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

    const int NUM_BALLS = 22;
    uniform vec2 u_ballPositions[NUM_BALLS];
    uniform float u_ballRadii[NUM_BALLS];

    // Smooth isosurface helper
    float smoothIso(float v, float iso, float k){
        return 1.0 - smoothstep(iso - k, iso + k, v);
    }

    // Totally new palette: electric lime/citron -> aqua -> indigo -> deep violet
    // Distinct from previous warm scheme
    vec3 palette(float t){
        t = clamp(t, 0.0, 1.0);
        vec3 c1 = vec3(0.80, 0.95, 0.10); // electric lime
        vec3 c2 = vec3(0.10, 0.95, 0.85); // aqua-mint
        vec3 c3 = vec3(0.25, 0.25, 0.95); // indigo
        vec3 c4 = vec3(0.55, 0.10, 0.75); // deep violet
        // segmented cubic blends
        if (t < 0.33) {
            float k = smoothstep(0.0, 0.33, t);
            return mix(c1, c2, k);
        } else if (t < 0.66) {
            float k = smoothstep(0.33, 0.66, t);
            return mix(c2, c3, k);
        } else {
            float k = smoothstep(0.66, 1.0, t);
            return mix(c3, c4, k);
        }
    }

    void main() {
        vec2 uv = gl_FragCoord.xy / u_resolution;

        // Accumulate scalar field and gradient
        float field = 0.0;
        vec2 grad = vec2(0.0);

        for (int i = 0; i < NUM_BALLS; i++) {
            vec2 p = u_ballPositions[i];
            float r = u_ballRadii[i];
            vec2 d = uv - p;
            float dist = max(length(d), 1e-4);
            float contrib = r / dist;
            field += contrib;
            grad += (-r) * d / (dist*dist*dist);
        }
        // mild time-based breathing to keep surface lively
        field += 0.15 * sin(u_time * 1.3 + uv.x*10.0 + uv.y*9.0);

        // Isosurface threshold controls blob size
        float iso = 7.0; // slightly higher to compensate for more blobs

        // Soft zones
        float body = smoothIso(field, iso, 0.32);
        float inner = smoothIso(field, iso + 1.0, 0.6);
        float glow  = smoothIso(field, iso - 0.9, 0.9);

        // Simple lighting from gradient
        vec2 n2 = normalize(grad + 1e-6);
        vec3 N = normalize(vec3(n2, 0.6));
        vec3 L = normalize(vec3(0.3, 0.5, 0.85));
        float diff = max(dot(N, L), 0.0);

        // Rim for nicer edge
        float rim = pow(1.0 - N.z, 2.0);

        // Color driven by field intensity
        float tcol = clamp((field - (iso - 1.8)) / 3.6, 0.0, 1.0);
        vec3 base = palette(tcol);

        // Compose
        vec3 col = vec3(0.0);
        col += base * (0.22 + 0.88*diff) * body;                 // lit body
        col += palette(min(1.0, tcol*1.2)) * inner * 0.65;         // core tint
        col += mix(vec3(0.10,0.95,0.85), vec3(0.55,0.10,0.75), 0.45) * glow * 0.6; // halo with new palette endpoints
        col += vec3(0.95, 1.0, 0.98) * rim * body * 0.28;          // subtle rim (cooler)

        // Background and blend
        vec3 bg = vec3(0.01, 0.02, 0.035); // slightly cooler/darker background
        vec3 finalCol = mix(bg, col, clamp(body + glow*0.75, 0.0, 1.0));

        gl_FragColor = vec4(max(finalCol, 0.0), 1.0);
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
    for (let i = 0; i < 22; i++) {
        metaballs.ballData.push({
            x: Math.random(),
            y: Math.random(),
            radius: Math.random() * 0.085 + 0.04,
            speedX: (Math.random() - 0.5) * 0.0014,
            speedY: (Math.random() - 0.5) * 0.0014
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

// --- Effect 14: Flame 2025 (Modern Stylized Fire) ---
const flame2025 = {};
flame2025.vsSource = quadVS;
flame2025.fsSource = `
    precision highp float;
    uniform vec2 u_resolution;
    uniform float u_time;

    // Hash/Noise helpers (value noise + fbm)
    float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453123); }
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
        mat2 m = mat2(1.6,1.2,-1.2,1.6);
        for(int i=0;i<5;i++){
            v += a * noise(p);
            p = m * p;
            a *= 0.5;
        }
        return v;
    }

    // Blackbody-ish palette from deep red -> orange -> yellow -> white
    vec3 firePalette(float t){
        t = clamp(t, 0.0, 1.0);
        float r = smoothstep(0.0, 0.25, t) + 0.65*t;
        float g = smoothstep(0.15, 0.9, t);
        float b = smoothstep(0.55, 1.0, t)*0.55;
        vec3 col = vec3(r, g, b);
        col = 1.0 - exp(-col*2.2);
        return col;
    }

    // Base emission profile: stronger at bottom, slower widening; clamp top width
    // Widened by ~20% to increase base footprint; includes horizontal source offset
    float baseSource(vec2 p){
        // Match the same horizontal offset used in main() so the emission aligns with visible plume
        float offsetX = 0.25;
        p.x -= offsetX;

        float y = p.y + 0.95;                  // base sits just below frame
        // Make base wider, and reduce widening rate with height
        float baseW = 0.26 * 1.20;             // +20% base width
        float widen = (0.36 * 1.20) * smoothstep(0.0, 1.0, clamp(y, 0.0, 1.0)); // +20% widen rate
        float w = baseW + widen;
        // Soft cap so top doesn't get too wide (also relaxed slightly)
        w = min(w, 0.52 * 1.10);
        float d = length(vec2(p.x / max(w, 0.001), max(y, 0.0)));
        return exp(-3.0 * d*d);
    }

    void main(){
        vec2 uv = gl_FragCoord.xy / u_resolution;

        // Center and shrink the flame footprint so it occupies less screen space
        // scale < 1.0 shrinks; tweak to adjust coverage
        float scale = 0.60;
        vec2 p = (uv * 2.0 - 1.0);
        p.x *= u_resolution.x / u_resolution.y;
        p /= scale;

        // Shift source horizontally to the right to avoid overlapping page text
        // Positive values move the flame to the right. Adjust offsetX as needed.
        float offsetX = 0.25; // ~25% of NDC after scaling
        p.x -= offsetX;

        float t = u_time;

        // Stronger upward advection, remove downward tendencies
        float rise = t * 1.05;

        // Lateral drift: make base broader, taper as it rises (prevents top overgrowth)
        // Slightly increase near-base outward factor for wider foot
        // Also bias drift a bit to the right so tongues lean with the new source
        float outward = (0.28*1.20 - 0.18 * smoothstep(-0.2, 0.9, p.y)) * p.y + 0.05;

        // Drive vertical samples with negative time bias to force upwards motion
        vec2 flowP = vec2(
            p.x + outward,
            // add slight upward push near base to feed the bottom body
            p.y + 0.32 * fbm(vec2(p.x*1.25, p.y*2.6 - rise*1.4)) + 0.06 * (1.0 - smoothstep(-0.9, -0.2, p.y))
        );

        // Domain warps: sample y with (-rise) to advect upwards
        vec2 warp1 = vec2(
            fbm(vec2(flowP.x*1.35 + 3.7, flowP.y*2.6 - rise*1.3)),
            fbm(vec2(flowP.x*1.75 + 9.1, flowP.y*2.9 - rise*1.6))
        );
        vec2 warp2 = vec2(
            fbm(vec2(flowP.x*3.1 + 17.0 + t*0.27, flowP.y*4.2 - rise*2.1)),
            fbm(vec2(flowP.x*2.3 + 27.0 - t*0.19, flowP.y*3.6 - rise*1.8))
        );

        // Anisotropic heat shimmer: vertical dominant; keep minimal near base for stability
        float baseMask = smoothstep(-0.85, -0.15, p.y);
        vec2 shimmer = 0.007 * baseMask * vec2(
            0.30 * sin((p.y - t*1.15)*28.0 + 2.2*fbm(p*4.8 - t)),
            1.00 * cos((p.x - t*1.05)*24.0 + 1.9*fbm(p*4.5 - t*0.7))
        );

        // Combine (slightly reduce warp amounts to keep footprint compact; stronger at base)
        float baseBoost = 0.08 * (1.0 - smoothstep(-0.9, -0.2, p.y));
        vec2 q = flowP + shimmer + (0.14+baseBoost)*(warp1 - 0.5) + (0.09+0.5*baseBoost)*(warp2 - 0.5);

        // Emission source from base, not centralizing
        float source = baseSource(q);

        // Upward density: bias more mass near the base, taper faster up high
        float dens = smoothstep(-0.95, 0.15, q.y) * (1.0 - 0.35 * smoothstep(0.0, 0.9, q.y));

        // Intensity via layered fbm with vertical advection (use -rise to bias upwards)
        float f1 = fbm(vec2(q.x*2.2, q.y*4.2 - rise*2.2));
        float f2 = fbm(vec2(q.x*1.1 + 4.0, q.y*2.0 - rise*1.5));
        float tongues = 0.6*f1 + 0.4*f2;
        // Small-scale turbulent flicker that travels up (phase uses -t)
        float flicker = 0.22 * sin(12.0*q.y + t*7.0 + 6.2831*fbm(q*3.5 - t*0.4));
        float intensity = clamp(tongues + flicker, 0.0, 1.0);
        intensity = pow(intensity, 1.20);
        // Prevent any appearance of downward pooling by damping intensity when q.y is low and decreasing
        float upBias = smoothstep(-0.95, 0.2, q.y);
        intensity *= upBias;

        // Temperature: increase base contribution to thicken bottom, and dampen top growth
        // Boost brightest area by ~20% via higher core feed (source) and mild intensity gain near base
        float heightDamp = 1.0 - 0.35 * smoothstep(0.1, 0.9, q.y);
        float baseGain = 1.32 * 1.20; // +20% brighter core feed from base emission
        float baseIntensityBoost = mix(1.20, 1.0, smoothstep(-0.6, 0.2, q.y)); // extra near base only
        float temp = (baseGain*source) + (0.95*dens*(intensity*baseIntensityBoost)*heightDamp);

        // Core/mid/glow shaping; emphasize base fullness and limit top bloom
        // Lower core threshold slightly so the brightest area spreads ~20% wider
        float core = smoothstep(0.64, 0.985, temp);
        float mid  = smoothstep(0.34, 0.80, temp) * (1.0 - 0.55*core);
        float glow = smoothstep(0.15, 0.44, temp) * (1.0 - 0.38*mid);

        // Colorization
        vec3 colCore = firePalette(0.88 + 0.12*temp);
        vec3 colMid  = firePalette(0.58 + 0.34*temp);
        vec3 colGlow = firePalette(0.32 + 0.28*temp);

        // Compose emissive
        vec3 col = vec3(0.0);
        col += colGlow * glow * 0.70;
        col += colMid  * mid  * 1.05;
        col += colCore * core * 1.65;

        // Subtle vertical-only haze to enhance upward motion
        float haze = 0.018 * sin( q.y*140.0 - t*8.0 + 3.0*fbm(q*6.0 + t) );
        col += vec3(haze);

        // Vignette: slightly relax top darkening so the cap doesn’t explode
        float vigX = 0.96 - 0.55*abs(p.x);
        float vigY = 0.985 - 0.30*max(0.0, p.y);
        col *= clamp(min(vigX, vigY), 0.20, 1.0);

        // Gentle scanline
        float scan = 0.020 * sin(gl_FragCoord.y * 3.14159 + t * 3.0);
        col += vec3(scan) * 0.025;

        gl_FragColor = vec4(max(col, 0.0), 1.0);
    }
`;
flame2025.init = () => {
    flame2025.program = createProgram(gl,
        createShader(gl, gl.VERTEX_SHADER, flame2025.vsSource),
        createShader(gl, gl.FRAGMENT_SHADER, flame2025.fsSource)
    );
    gl.useProgram(flame2025.program);
    flame2025.positionAttributeLocation = gl.getAttribLocation(flame2025.program, 'a_position');
    flame2025.resolutionUniformLocation = gl.getUniformLocation(flame2025.program, 'u_resolution');
    flame2025.timeUniformLocation = gl.getUniformLocation(flame2025.program, 'u_time');
};
flame2025.draw = (time) => {
    gl.useProgram(flame2025.program);
    gl.uniform2f(flame2025.resolutionUniformLocation, canvas.width, canvas.height);
    gl.uniform1f(flame2025.timeUniformLocation, time / 1000.0);
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.enableVertexAttribArray(flame2025.positionAttributeLocation);
    gl.vertexAttribPointer(flame2025.positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
};
flame2025.name = 'Flame 2025';
effects.push(flame2025);

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

// --- Effect: Particle Vector Field (brand-new GPU particle flow) ---
const particleVectorField = {};
particleVectorField.vsSource = `
    precision mediump float; // keep precision explicit and consistent
    attribute float a_id;
    uniform vec2 u_resolution;
    uniform float u_time;

    // Hash helpers
    float hash(float n){ return fract(sin(n)*43758.5453123); }
    vec2  hash2(float n){ return fract(sin(vec2(n, n+1.23))*vec2(43758.5453, 22578.145912)); }

    // 2D rotation
    mat2 rot(float a){ return mat2(cos(a), -sin(a), sin(a), cos(a)); }

    // Pseudo curl-like flow built from simple sin/cos warps (fast, branchless)
    vec2 flow(vec2 p, float t){
        float s1 = sin(p.y*2.3 + t*0.9);
        float c1 = cos(p.x*2.7 - t*1.1);
        float s2 = sin(p.x*3.7 - p.y*1.9 + t*0.6);
        float c2 = cos(p.y*3.1 + p.x*2.2 - t*0.5);
        vec2 v = vec2(s1 + 0.6*s2, -c1 + 0.6*c2);
        // swirl
        float a = 0.35*sin(t*0.27) + 0.2*sin(t*0.13);
        v = rot(a)*v;
        return v;
    }

    void main(){
        // Each particle gets a deterministic seed from its id
        float id = a_id;
        vec2 seed = hash2(id*13.37 + 7.1);

        // Make the field time-coherent: remove lifecycle reset and use a slow global time
        float T = u_time * 0.25; // slow everything down
        float tLocal = T;

        // Spawn base in [-1,1] around screen center with stratified jitter
        vec2 spawn = (seed*2.0 - 1.0);
        spawn.x *= u_resolution.x / u_resolution.y; // aspect
        spawn *= 0.95;

        // Integrate simple flow (semi-analytic trail param)
        vec2 p = spawn;
        float dt = 0.03;              // smaller step for smoother, slower advection
        float t = tLocal + hash(id+91.0)*3.0; // small per-particle dephase, constant over time
        for (int i=0; i<28; i++){
            vec2 v = flow(p*1.2 + seed*2.0, t*0.45); // reduce flow temporal speed
            p += v * dt * (0.35 + 0.45*hash(id + float(i)*17.0)); // overall slower motion
            t += dt;
        }

        // Subtle drift so field breathes (slower + smaller)
        p += 0.04*vec2(sin(T*0.12 + seed.x*6.0), cos(T*0.10 + seed.y*7.0));

        // Fade near edges to avoid harsh clipping
        float ar = u_resolution.x / u_resolution.y;
        vec2 q = p; q.x /= ar;
        float edge = smoothstep(1.2, 0.85, max(abs(q.x), abs(q.y)));

        // Convert NDC to clipspace
        vec2 clip = p;
        clip.x /= ar;

        gl_Position = vec4(clip, 0.0, 1.0);

        // Size varies with edge fade with much gentler flicker
        float size = 1.4 + 1.2*edge + 0.3*sin(id*12.73 + T*2.0);
        gl_PointSize = max(1.0, size);
    }
`;

particleVectorField.fsSource = `
    precision mediump float;
    uniform float u_time;
    // Soft round points
    void main(){
        vec2 uv = gl_PointCoord*2.0 - 1.0;
        float r2 = dot(uv, uv);
        float alpha = smoothstep(1.0, 0.0, r2);

        // Dim overall intensity and reduce pulse amplitude
        vec3 baseA = vec3(0.035, 0.63, 0.66); // dimmed cyan
        vec3 baseB = vec3(0.57, 0.12, 0.57);  // dimmed magenta
        vec3 col = mix(baseA, baseB, smoothstep(0.0, 1.0, alpha));

        // Softer, slower pulse
        col *= 0.94 + 0.06 * sin(u_time * 1.6);

        // Lower maximum alpha for less additive bloom
        float a = alpha * 0.55;

        gl_FragColor = vec4(col, a);
    }
`;

particleVectorField.init = () => {
    // Create program
    particleVectorField.program = createProgram(gl,
        createShader(gl, gl.VERTEX_SHADER, particleVectorField.vsSource),
        createShader(gl, gl.FRAGMENT_SHADER, particleVectorField.fsSource)
    );
    gl.useProgram(particleVectorField.program);

    // Attributes/uniforms
    particleVectorField.aId = gl.getAttribLocation(particleVectorField.program, 'a_id');
    particleVectorField.uRes = gl.getUniformLocation(particleVectorField.program, 'u_resolution');
    particleVectorField.uTime = gl.getUniformLocation(particleVectorField.program, 'u_time');

    // Allocate particle ids once
    const COUNT = 180000; // high count, but simple shader keeps it smooth
    particleVectorField.count = COUNT;
    const ids = new Float32Array(COUNT);
    for (let i=0;i<COUNT;i++) ids[i] = i;

    particleVectorField.idBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, particleVectorField.idBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, ids, gl.STATIC_DRAW);

    // Blending for additive neon trails
    particleVectorField.prevState = {
        blend: gl.isEnabled(gl.BLEND),
        depthTest: gl.isEnabled(gl.DEPTH_TEST)
    };
};

particleVectorField.draw = (time) => {
    gl.useProgram(particleVectorField.program);
    gl.uniform2f(particleVectorField.uRes, canvas.width, canvas.height);
    gl.uniform1f(particleVectorField.uTime, time / 1000.0);

    // Render state for additive particles (use softer additive to reduce brightness)
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    // Keep additive feel but dampen overlaps by scaling src alpha down in shader.
    // Optionally, change to premultiplied style for even softer result:
    // gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);

    gl.bindBuffer(gl.ARRAY_BUFFER, particleVectorField.idBuffer);
    gl.enableVertexAttribArray(particleVectorField.aId);
    gl.vertexAttribPointer(particleVectorField.aId, 1, gl.FLOAT, false, 0, 0);

    gl.drawArrays(gl.POINTS, 0, particleVectorField.count);

    // restore default for safety (other effects set their own, but keep tidy)
    gl.disableVertexAttribArray(particleVectorField.aId);
    gl.disable(gl.BLEND);
    gl.enable(gl.DEPTH_TEST);
};
particleVectorField.name = 'Particle Vector Field';
effects.push(particleVectorField);

// Removed per-effect order mutation to avoid mid-registration reorders

// --- Effect: Hex Lattice Warp (brand-new) ---
const hexLatticeWarp = {};
hexLatticeWarp.vsSource = quadVS;
hexLatticeWarp.fsSource = `
    precision highp float;
    uniform vec2 u_resolution;
    uniform float u_time;

    // Compute hex cell edge intensity via nearest-center approximation
    float hexEdge(vec2 p, float scale){
        vec2 hp = p / scale;

        // axial rounding to nearest hex center in axial-coord space
        float q = (2.0/3.0) * hp.x;
        float r = (-1.0/3.0) * hp.x + (0.57735026919) * hp.y; // 1/sqrt(3)
        float rq = floor(q + 0.5);
        float rr = floor(r + 0.5);

        // convert axial back to cartesian center
        vec2 base = vec2(rq + rr*0.5, rr*0.86602540378);

        // vector from center to point
        vec2 l = hp - base;

        // distance to the three orientations of hex edges (using apothem axes)
        float a = abs(l.x);
        float b = abs(0.5*l.x + 0.86602540378*l.y);
        float c = abs(-0.5*l.x + 0.86602540378*l.y);

        float edge = max(a, max(b, c));
        // map approx center->edge to 0..1 and smooth
        float ed = smoothstep(0.70, 0.86, edge);
        return ed;
    }

    // Hash/noise helpers
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
        mat2 m = mat2(1.6,1.2,-1.2,1.6);
        for(int i=0;i<4;i++){
            v += a * noise(p);
            p = m * p;
            a *= 0.5;
        }
        return v;
    }

    // Palette: teal -> purple -> amber
    vec3 palette(float t){
        vec3 c1 = vec3(0.05, 0.9, 0.9);
        vec3 c2 = vec3(0.65, 0.2, 0.95);
        vec3 c3 = vec3(1.0, 0.7, 0.15);
        vec3 a = mix(c1, c2, smoothstep(0.0, 0.5, t));
        vec3 b = mix(c2, c3, smoothstep(0.5, 1.0, t));
        return mix(a, b, step(0.5, t));
    }

    void main(){
        vec2 uv = gl_FragCoord.xy / u_resolution;
        vec2 p = uv * 2.0 - 1.0;
        p.x *= u_resolution.x / u_resolution.y;

        float t = u_time;

        // three parallaxed layers with domain warp
        float s1 = 0.75;
        float s2 = 0.45;
        float s3 = 0.25;

        vec2 w1 = 0.18 * vec2(
            fbm(p*1.1 + vec2(t*0.21, -t*0.17)),
            fbm(p*1.0 + vec2(-t*0.19, t*0.23))
        );
        vec2 w2 = 0.28 * vec2(
            fbm(p*2.3 + vec2(-t*0.11, t*0.15)),
            fbm(p*2.1 + vec2(t*0.09, -t*0.07))
        );
        vec2 w3 = 0.42 * vec2(
            fbm(p*3.7 + vec2(t*0.035, t*0.027)),
            fbm(p*3.2 + vec2(-t*0.031, -t*0.044))
        );

        float ang = 0.12 * t;
        mat2 R = mat2(cos(ang), -sin(ang), sin(ang), cos(ang));
        vec2 q1 = R * (p + w1 * 0.6);
        vec2 q2 = R * (p*1.3 + w2);
        vec2 q3 = R * (p*1.7 + w3);

        float e1 = hexEdge(q1, s1);
        float e2 = hexEdge(q2, s2);
        float e3 = hexEdge(q3, s3);

        float edgeNeon = pow(smoothstep(0.75, 1.0, e1)*0.8 + smoothstep(0.72, 1.0, e2)*0.7 + smoothstep(0.70, 1.0, e3)*0.6, 1.4);

        float cellPulse = 0.0;
        cellPulse += 0.55 * (0.5 + 0.5*sin(8.0*fbm(q1*1.2 + t*0.3) + t*1.1));
        cellPulse += 0.35 * (0.5 + 0.5*sin(10.0*fbm(q2*1.7 - t*0.2) + t*0.9));
        cellPulse += 0.25 * (0.5 + 0.5*sin(12.0*fbm(q3*2.0 + t*0.1) + t*0.7));

        float vig = 0.92 - 0.55 * dot(p,p);
        float scan = 0.025 * sin(gl_FragCoord.y * 3.14159 + t * 3.2);

        float hue = fract(0.5 + 0.3*sin(t*0.37) + 0.2*sin(p.x*1.8 + p.y*1.3 + t*0.6));
        vec3 base = palette(hue);

        vec3 edgeCol = mix(vec3(0.0, 0.9, 1.0), vec3(0.9, 0.2, 1.0), 0.5 + 0.5*sin(t*0.5));
        vec3 coreCol = mix(vec3(1.0, 0.7, 0.15), base, 0.4);

        vec3 col = vec3(0.0);
        col += edgeCol * edgeNeon * 0.95;
        col += coreCol * cellPulse * 0.55;
        col = mix(col, base, 0.20);

        col *= clamp(vig, 0.25, 1.0);
        col += vec3(scan) * 0.03;

        gl_FragColor = vec4(max(col, 0.0), 1.0);
    }
`;

hexLatticeWarp.init = () => {
    hexLatticeWarp.program = createProgram(gl,
        createShader(gl, gl.VERTEX_SHADER, hexLatticeWarp.vsSource),
        createShader(gl, gl.FRAGMENT_SHADER, hexLatticeWarp.fsSource)
    );
    gl.useProgram(hexLatticeWarp.program);
    hexLatticeWarp.positionAttributeLocation = gl.getAttribLocation(hexLatticeWarp.program, 'a_position');
    hexLatticeWarp.resolutionUniformLocation = gl.getUniformLocation(hexLatticeWarp.program, 'u_resolution');
    hexLatticeWarp.timeUniformLocation = gl.getUniformLocation(hexLatticeWarp.program, 'u_time');
};
hexLatticeWarp.draw = (time) => {
    gl.useProgram(hexLatticeWarp.program);
    gl.uniform2f(hexLatticeWarp.resolutionUniformLocation, canvas.width, canvas.height);
    gl.uniform1f(hexLatticeWarp.timeUniformLocation, time / 1000.0);
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.enableVertexAttribArray(hexLatticeWarp.positionAttributeLocation);
    gl.vertexAttribPointer(hexLatticeWarp.positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
};
hexLatticeWarp.name = 'Hex Lattice Warp';
effects.push(hexLatticeWarp);

// Removed per-effect order mutation to avoid mid-registration reorders

// --- Effect: Ember Flock (brand-new warm ember swarm) ---
const emberFlock = {};
emberFlock.vsSource = quadVS;
emberFlock.fsSource = `
    precision highp float;
    uniform vec2 u_resolution;
    uniform float u_time;

    float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453123); }
    float noise(vec2 p){
        vec2 i = floor(p);
        vec2 f = fract(p);
        f = f*f*(3.0-2.0*f);
        float a = hash(i);
        float b = hash(i+vec2(1.0,0.0));
        float c = hash(i+vec2(0.0,1.0));
        float d = hash(i+vec2(1.0,1.0));
        return mix(mix(a,b,f.x), mix(c,d,f.x), f.y);
    }
    float fbm(vec2 p){
        float v = 0.0;
        float a = 0.5;
        mat2 m = mat2(1.6,1.2,-1.2,1.6);
        for (int i=0;i<5;i++){
            v += a * noise(p);
            p = m * p;
            a *= 0.5;
        }
        return v;
    }

    // Warm blackbody-like palette: deep maroon -> crimson -> orange -> amber
    vec3 emberPalette(float t){
        t = clamp(t, 0.0, 1.0);
        vec3 c1 = vec3(0.14, 0.03, 0.02);
        vec3 c2 = vec3(0.75, 0.16, 0.06);
        vec3 c3 = vec3(1.00, 0.55, 0.12);
        vec3 c4 = vec3(1.00, 0.86, 0.30);
        vec3 a = mix(c1, c2, smoothstep(0.00, 0.35, t));
        vec3 b = mix(c3, c4, smoothstep(0.35, 1.00, t));
        return mix(a, b, smoothstep(0.25, 0.95, t));
    }

    float softKernel(vec2 p, float r){
        float d2 = dot(p,p);
        return exp(-d2 / (r*r));
    }

    vec3 renderEmbers(vec2 uv, float t){
        vec2 p = uv;
        p.x *= u_resolution.x / u_resolution.y;

        // Global turbulent flow
        vec2 flow = vec2(
            fbm(p*0.7 + vec2( t*0.18, -t*0.12 )),
            fbm(p*0.7 + vec2(-t*0.15,  t*0.14 ))
        );
        float ang = 0.35 * sin(t*0.25) + 0.25*sin(t*0.11);
        mat2 R = mat2(cos(ang), -sin(ang), sin(ang), cos(ang));
        vec2 q = R * (p + (flow - 0.5)*0.9);

        float glow = 0.0;
        float core = 0.0;

        // Layer 1 (coarse)
        {
            vec2 gp = q * 3.0;
            vec2 cell = floor(gp);
            vec2 f = fract(gp);
            for (int j=0;j<2;j++){
                for (int i=0;i<2;i++){
                    vec2 o = vec2(float(i), float(j));
                    vec2 id = cell + o;
                    float s = hash(id);
                    vec2 jitter = vec2(sin(s*87.0), cos(s*113.0));
                    vec2 center = (o + 0.5 + 0.35*jitter) / 3.0;
                    center += 0.08*vec2(sin(t*0.9 + s*17.0), cos(t*0.8 + s*23.0));
                    vec2 d = f - (center);
                    float k = softKernel(d, 0.22);
                    glow += k * (0.8 + 0.2*sin(t*2.0 + s*31.0));
                    core += k*k;
                }
            }
        }
        // Layer 2 (medium)
        {
            vec2 gp = q * 6.0 + 2.7;
            vec2 cell = floor(gp);
            vec2 f = fract(gp);
            for (int j=0;j<2;j++){
                for (int i=0;i<2;i++){
                    vec2 o = vec2(float(i), float(j));
                    vec2 id = cell + o;
                    float s = hash(id + 101.3);
                    vec2 jitter = vec2(sin(s*55.0), cos(s*71.0));
                    vec2 center = (o + 0.5 + 0.45*jitter) / 6.0;
                    center += 0.12*vec2(sin(t*1.2 + s*13.0), cos(t*1.15 + s*19.0));
                    vec2 d = f - (center);
                    float k = softKernel(d, 0.18);
                    glow += k * (0.7 + 0.3*sin(t*2.6 + s*41.0));
                    core += k*k*1.3;
                }
            }
        }
        // Layer 3 (fine)
        {
            vec2 gp = q * 12.0 - 1.9;
            vec2 cell = floor(gp);
            vec2 f = fract(gp);
            for (int j=0;j<2;j++){
                for (int i=0;i<2;i++){
                    vec2 o = vec2(float(i), float(j));
                    vec2 id = cell + o;
                    float s = hash(id + 407.7);
                    vec2 jitter = vec2(sin(s*93.0), cos(s*131.0));
                    vec2 center = (o + 0.5 + 0.55*jitter) / 12.0;
                    center += 0.16*vec2(sin(t*1.7 + s*29.0), cos(t*1.6 + s*37.0));
                    vec2 d = f - (center);
                    float k = softKernel(d, 0.13);
                    glow += k * (0.55 + 0.45*sin(t*3.3 + s*73.0));
                    core += k*k*1.6;
                }
            }
        }

        glow = clamp(glow*0.9, 0.0, 3.0);
        core = clamp(core*1.4, 0.0, 3.0);

        float heat = clamp(0.35*glow + 0.65*core, 0.0, 1.5);
        float tcol = smoothstep(0.0, 1.2, heat);

        vec3 col = emberPalette(tcol);
        vec3 coreCol = emberPalette(min(1.0, tcol*1.25));
        col = mix(col, coreCol, clamp(core*0.5, 0.0, 0.7));

        float flick = 0.92 + 0.08*sin(t*7.0 + fbm(p*4.0 + t)*6.2831);
        col *= flick;

        float vig = 0.92 - 0.55*dot(uv, uv);
        col *= clamp(vig, 0.25, 1.0);

        float scan = 0.018 * sin(gl_FragCoord.y * 3.14159 + t * 3.0);
        col += vec3(scan) * 0.015;

        return col;
    }

    void main(){
        vec2 uv = gl_FragCoord.xy / u_resolution;
        vec2 p = uv * 2.0 - 1.0;
        float t = u_time;
        vec3 col = renderEmbers(p, t);
        gl_FragColor = vec4(max(col, 0.0), 1.0);
    }
`;
emberFlock.init = () => {
    emberFlock.program = createProgram(gl,
        createShader(gl, gl.VERTEX_SHADER, emberFlock.vsSource),
        createShader(gl, gl.FRAGMENT_SHADER, emberFlock.fsSource)
    );
    gl.useProgram(emberFlock.program);
    emberFlock.positionAttributeLocation = gl.getAttribLocation(emberFlock.program, 'a_position');
    emberFlock.resolutionUniformLocation = gl.getUniformLocation(emberFlock.program, 'u_resolution');
    emberFlock.timeUniformLocation = gl.getUniformLocation(emberFlock.program, 'u_time');
};
emberFlock.draw = (time) => {
    gl.useProgram(emberFlock.program);
    gl.uniform2f(emberFlock.resolutionUniformLocation, canvas.width, canvas.height);
    gl.uniform1f(emberFlock.timeUniformLocation, time / 1000.0);
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.enableVertexAttribArray(emberFlock.positionAttributeLocation);
    gl.vertexAttribPointer(emberFlock.positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
};
emberFlock.name = 'Ember Flock';
effects.push(emberFlock);

// Removed IIFE that forced Ember Flock to index 0

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

// --- Effect: Neon Smoke Portal (brand-new raymarch) ---
const neonSmokePortal = {};
neonSmokePortal.vsSource = quadVS;
neonSmokePortal.fsSource = `
    precision highp float;
    uniform vec2 u_resolution;
    uniform float u_time;

    // Hash/Noise
    float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453123); }
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
    float fbm2(vec2 p){
        // cheaper fbm: 3 octaves
        float v = 0.0;
        float a = 0.5;
        mat2 m = mat2(1.6,1.2,-1.2,1.6);
        for(int i=0;i<3;i++){
            v += a * noise(p);
            p = m * p;
            a *= 0.5;
        }
        return v;
    }

    // Signed distance: Torus
    float sdTorus(vec3 p, vec2 t){
        vec2 q = vec2(length(p.xz) - t.x, p.y);
        return length(q) - t.y;
    }

    // Simple 2D rotators
    mat2 rot2(float a){ return mat2(cos(a), -sin(a), sin(a), cos(a)); }

    // Palette (neon)
    vec3 palette(float h){
        vec3 a = vec3(0.05,0.90,0.95); // cyan
        vec3 b = vec3(0.95,0.20,0.95); // magenta
        vec3 c = vec3(1.00,0.75,0.20); // amber
        return mix(mix(a,b,smoothstep(0.0,0.6,h)), c, smoothstep(0.6,1.0,h));
    }

    // Scene distance: torus with domain warp for smoky edges
    float map(vec3 p, float t, out float edgeMask){
        // 3-axis rotation
        float ay = 0.4*sin(t*0.40) + 0.9*sin(t*0.17);
        float ax = 0.6*sin(t*0.33) + 0.4*sin(t*0.21);
        float az = 0.5*sin(t*0.27) - 0.6*sin(t*0.13);
        // apply in sequence: Y, X, Z
        p.xz = rot2(ay) * p.xz;  // around Y
        p.yz = rot2(ax) * p.yz;  // around X
        p.xy = rot2(az) * p.xy;  // around Z

        // Domain noise warp (cheaper & reused)
        float w1 = fbm2(p.xy*0.7 + t*0.15);
        float w2 = fbm2(p.zy*0.8 - t*0.11);
        float w = w1 + w2;
        p += 0.28 * (w - 1.0) * normalize(vec3(p.x, p.y*0.4, p.z) + 0.0001);

        float d = sdTorus(p, vec2(1.25, 0.28));

        // Edge mask for neon accent
        edgeMask = smoothstep(0.35, 0.0, abs(d));
        return d;
    }

    // Raymarch
    vec4 raymarch(vec3 ro, vec3 rd, float t){
        // jitter start to reduce banding, allows fewer steps
        float seed = dot(gl_FragCoord.xy, vec2(0.13,0.71));
        float jitter = fract(sin(seed)*43758.5453);
        float total = jitter * 0.06;

        vec3 col = vec3(0.0);
        float accGlow = 0.0;

        // Coarse hue base and parameters
        float hueBase = 0.5 + 0.5*sin(t*0.45);
        float maxDist = 6.0;

        // Fewer steps + stronger adaptive stepping
        for(int i=0;i<44;i++){
            vec3 pos = ro + rd * total;
            float edgeMask;
            float d = map(pos, t, edgeMask);

            // Density from SDF
            float dens = exp(-6.5*abs(d));
            // cheaper modulation
            float n = fbm2(pos.xz*0.75 + t*0.42);
            dens *= 0.6 + 0.4*sin(4.7*n + t*1.8);

            // Color
            float hue = clamp(hueBase + 0.3*sin(0.55*pos.x + 0.85*pos.y + 0.65*pos.z), 0.0, 1.0);
            vec3 c = palette(hue);
            c += vec3(0.2, 0.8, 1.0) * (edgeMask*edgeMask) * 0.5;

            // Accumulate
            float stepLen = 0.055;
            float contrib = dens * stepLen * (0.6 + 0.4*edgeMask);
            col += c * contrib;
            accGlow += contrib * 0.85;

            // Early outs
            if(accGlow > 0.80) break;

            // Adaptive step: larger as we go, and when away from surface
            float adv = max(0.035, min(0.20, abs(d)*0.75 + total*0.03));
            total += adv;
            if(total > maxDist) break;
        }

        vec3 bg = vec3(0.01, 0.02, 0.04);
        vec3 finalCol = mix(bg, col, clamp(accGlow*1.25, 0.0, 1.0));
        finalCol += vec3(accGlow*0.5); // bloom-ish reduced

        return vec4(finalCol, 1.0);
    }

    void main(){
        vec2 uv = gl_FragCoord.xy / u_resolution;
        vec2 p = uv * 2.0 - 1.0;
        p.x *= u_resolution.x / u_resolution.y;

        float t = u_time * 0.9;

        // Camera swirl
        float camR = 3.5;
        float ca = 0.6 * sin(t*0.33);
        vec3 ro = vec3(camR*cos(ca), 0.5 + 0.3*sin(t*0.2), camR*sin(ca));
        vec3 ta = vec3(0.0, 0.0, 0.0);

        // Build camera basis
        vec3 ww = normalize(ta - ro);
        vec3 uu = normalize(cross(vec3(0.0,1.0,0.0), ww));
        vec3 vv = cross(ww, uu);

        // Ray
        float fov = 1.2;
        vec3 rd = normalize(uu*p.x + vv*p.y + ww*fov);

        vec4 col = raymarch(ro, rd, t);

        // Film grain (cheap)
        float grain = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898,78.233)) + t*37.2)*43758.5453);
        col.rgb += (grain-0.5)*0.010;

        // Vignette
        float vig = 0.92 - 0.5*dot(p,p);
        col.rgb *= clamp(vig, 0.25, 1.0);

        // Subtle scanlines
        float scan = 0.012 * sin(gl_FragCoord.y * 3.14159 + t * 3.1);
        col.rgb += vec3(scan) * 0.012;

        gl_FragColor = vec4(max(col.rgb, 0.0), 1.0);
    }
`;
neonSmokePortal.init = () => {
    neonSmokePortal.program = createProgram(gl,
        createShader(gl, gl.VERTEX_SHADER, neonSmokePortal.vsSource),
        createShader(gl, gl.FRAGMENT_SHADER, neonSmokePortal.fsSource)
    );
    gl.useProgram(neonSmokePortal.program);
    neonSmokePortal.positionAttributeLocation = gl.getAttribLocation(neonSmokePortal.program, 'a_position');
    neonSmokePortal.resolutionUniformLocation = gl.getUniformLocation(neonSmokePortal.program, 'u_resolution');
    neonSmokePortal.timeUniformLocation = gl.getUniformLocation(neonSmokePortal.program, 'u_time');
};
neonSmokePortal.draw = (time) => {
    gl.useProgram(neonSmokePortal.program);
    gl.uniform2f(neonSmokePortal.resolutionUniformLocation, canvas.width, canvas.height);
    gl.uniform1f(neonSmokePortal.timeUniformLocation, time / 1000.0);
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.enableVertexAttribArray(neonSmokePortal.positionAttributeLocation);
    gl.vertexAttribPointer(neonSmokePortal.positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
};
neonSmokePortal.name = 'Neon Smoke Portal';
effects.push(neonSmokePortal);

// --- Effect: Hyperspace Glyphs (brand-new, neon runes in layered parallax) ---
const hyperspaceGlyphs = {};
hyperspaceGlyphs.vsSource = quadVS;
hyperspaceGlyphs.fsSource = `
    precision mediump float;
    uniform vec2 u_resolution;
    uniform float u_time;

    float hash(float n){ return fract(sin(n)*43758.5453123); }
    float hash2(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7)))*43758.5453); }
    mat2 rot(float a){ return mat2(cos(a), -sin(a), sin(a), cos(a)); }

    float sdBox(vec2 p, vec2 b){
        vec2 d = abs(p) - b;
        return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
    }
    float sdCircle(vec2 p, float r){ return length(p) - r; }
    float sdTriangleEq(vec2 p){
        const float k = 0.57735026919;
        p.x = abs(p.x);
        return max(p.y, k*p.x - 0.5);
    }
    float sdRing(vec2 p, float r, float t){
        return abs(length(p)-r) - t;
    }

    float smin(float a, float b, float k){
        float h = clamp(0.5 + 0.5*(b-a)/k, 0.0, 1.0);
        return mix(b, a, h) - k*h*(1.0-h);
    }

    float runeSDF(vec2 p, float idx){
        float d = 1e9;
        float t = idx;

        float frame = sdBox(p, vec2(0.90, 0.90)) - 0.10;
        d = min(d, frame);

        float variant = fract(t*0.73);
        if (variant < 0.33){
            float c = sdRing(p, 0.45, 0.06);
            float v = sdBox(p, vec2(0.06, 0.65));
            float h = sdBox(p.yx, vec2(0.06, 0.65));
            d = smin(d, c, 0.12);
            d = smin(d, v, 0.12);
            d = smin(d, h, 0.12);
        } else if (variant < 0.66){
            vec2 q = rot(0.523599)*p;
            float tri = sdTriangleEq(q*1.1);
            float orb = sdCircle(p, 0.28);
            d = smin(d, tri, 0.10);
            d = smin(d, abs(orb)-0.06, 0.10);
        } else {
            float r1 = sdRing(p, 0.42, 0.05);
            float r2 = sdRing(p, 0.24, 0.05);
            vec2 q = rot(0.35)*p;
            float bar1 = sdBox(q, vec2(0.65, 0.06));
            q = rot(-0.7)*q;
            float bar2 = sdBox(q, vec2(0.55, 0.06));
            d = smin(d, r1, 0.10);
            d = smin(d, r2, 0.10);
            d = smin(d, bar1, 0.08);
            d = smin(d, bar2, 0.08);
        }

        vec2 rp = abs(p) - vec2(0.82, 0.82);
        float tick = sdBox(rp, vec2(0.14, 0.02));
        vec2 rp2 = vec2(p.x, -p.y); rp2 = abs(rp2) - vec2(0.82, 0.82);
        float tick2 = sdBox(rp2, vec2(0.14, 0.02));
        vec2 rp3 = vec2(-p.x, p.y); rp3 = abs(rp3) - vec2(0.82, 0.82);
        float tick3 = sdBox(rp3, vec2(0.14, 0.02));
        vec2 rp4 = -p; rp4 = abs(rp4) - vec2(0.82, 0.82);
        float tick4 = sdBox(rp4, vec2(0.14, 0.02));
        d = smin(d, min(min(tick, tick2), min(tick3, tick4)), 0.08);

        return d;
    }

    vec3 palette(float h){
        vec3 c1 = vec3(0.05, 0.90, 0.95);
        vec3 c2 = vec3(0.95, 0.20, 0.95);
        vec3 c3 = vec3(1.00, 0.75, 0.20);
        return mix(mix(c1, c2, smoothstep(0.0, 0.6, h)), c3, smoothstep(0.6, 1.0, h));
    }

    vec3 renderGlyph(vec2 uv, float seed, float t, out float alpha){
        float baseIdx = floor(seed*997.0);
        float ang = 0.6*sin(seed*12.3 + t*0.7) + 0.4*sin(seed*7.1 - t*0.53);
        vec2 p = rot(ang) * uv;

        float dec = 0.85 + 0.15*sin(t*3.0 + seed*23.7);
        p *= dec;

        float d = runeSDF(p, baseIdx);

        float line = 1.0 - smoothstep(0.020, 0.030, abs(d));
        float glow = 1.0 - smoothstep(0.14, 0.46, abs(d));

        float hue = fract(0.47 + 0.33*sin(seed*19.0) + 0.25*sin(t*0.35 + seed*5.0));
        vec3 col = palette(hue);

        float flick = 0.6 + 0.4*sin(t*6.0 + seed*11.1);
        col *= 0.9 + 0.1*flick;

        vec3 c = vec3(0.0);
        c += col * line * 1.15;
        c += col * glow * 0.50;

        alpha = clamp(line + glow*0.6, 0.0, 1.0);
        return c;
    }

    void tileSpace(in vec2 p, float scale, out vec2 local, out vec2 id){
        vec2 gp = p*scale;
        vec2 g = floor(gp);
        vec2 f = fract(gp);
        local = (f*2.0 - 1.0);
        id = g;
    }

    void main(){
        vec2 R = u_resolution;
        vec2 uv = (gl_FragCoord.xy / R)*2.0 - 1.0;
        uv.x *= R.x / R.y;

        float t = u_time;

        vec3 accum = vec3(0.0);

        vec2 cam = 0.25*vec2(sin(t*0.2), cos(t*0.17));

        for (int i=0;i<4;i++){
            float fi = float(i);
            float depth = mix(1.8, 0.65, fi/3.0);
            float scale = mix(3.5, 9.5, fi/3.0);
            float speed = mix(0.08, 0.35, fi/3.0);

            vec2 p = uv;
            p += cam * (0.15 + 0.25*fi);
            p += vec2(0.35*sin(t*speed*0.9 + fi*1.7), 0.35*cos(t*speed*1.1 - fi*1.37));
            p = rot(0.07*sin(t*0.25 + fi))*p;

            vec2 local, id;
            tileSpace(p, scale, local, id);

            float vig = 1.0 - 0.35*dot(uv, uv);

            vec3 layerColor = vec3(0.0);
            float layerAlpha = 0.0;

            for (int oy=-1; oy<=0; oy++){
                for (int ox=-1; ox<=0; ox++){
                    vec2 off = vec2(float(ox), float(oy));
                    vec2 lid = id + off;
                    float s = hash2(lid);
                    vec2 jitter = vec2(sin(s*57.0), cos(s*91.0))*0.18;
                    vec2 l = local - off*2.0 + jitter;
                    l.x *= 0.9;

                    float a;
                    vec3 c = renderGlyph(l/depth, s, t, a);

                    float w = exp(-0.9*dot(l, l));
                    layerColor += c * a * w;
                    layerAlpha += a * w;
                }
            }

            layerColor /= max(0.0001, layerAlpha);
            layerAlpha = clamp(layerAlpha, 0.0, 1.0);

            float hueL = fract(0.2*fi + 0.15*sin(t*0.23 + fi));
            vec3 tint = palette(hueL);
            layerColor = mix(layerColor, layerColor * tint, 0.30);
            layerColor *= vig;

            float shutter = 0.7 + 0.3*sin(t*2.0 + fi*1.3);
            layerAlpha *= shutter;

            accum = mix(accum, layerColor, clamp(layerAlpha*0.65, 0.0, 1.0));
        }

        float scan = 0.016 * sin(gl_FragCoord.y * 3.14159 + t * 3.0);
        float grain = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898,78.233)) + t*41.2)*43758.5453) - 0.5;
        accum += vec3(scan) * 0.012 + vec3(grain) * 0.008;

        float centerGlow = exp(-2.0*dot(uv,uv));
        accum += vec3(0.08, 0.20, 0.30) * centerGlow * 0.45;

        gl_FragColor = vec4(max(accum, 0.0), 1.0);
    }
`;

hyperspaceGlyphs.init = () => {
    hyperspaceGlyphs.program = createProgram(gl,
        createShader(gl, gl.VERTEX_SHADER, hyperspaceGlyphs.vsSource),
        createShader(gl, gl.FRAGMENT_SHADER, hyperspaceGlyphs.fsSource)
    );
    gl.useProgram(hyperspaceGlyphs.program);
    hyperspaceGlyphs.positionAttributeLocation = gl.getAttribLocation(hyperspaceGlyphs.program, 'a_position');
    hyperspaceGlyphs.resolutionUniformLocation = gl.getUniformLocation(hyperspaceGlyphs.program, 'u_resolution');
    hyperspaceGlyphs.timeUniformLocation = gl.getUniformLocation(hyperspaceGlyphs.program, 'u_time');
};

hyperspaceGlyphs.draw = (time) => {
    gl.useProgram(hyperspaceGlyphs.program);
    gl.uniform2f(hyperspaceGlyphs.resolutionUniformLocation, canvas.width, canvas.height);
    gl.uniform1f(hyperspaceGlyphs.timeUniformLocation, time / 1000.0);

    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.enableVertexAttribArray(hyperspaceGlyphs.positionAttributeLocation);
    gl.vertexAttribPointer(hyperspaceGlyphs.positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
};
hyperspaceGlyphs.name = 'Hyperspace Glyphs';
effects.push(hyperspaceGlyphs);

// --- Effect: Neon Parallax City (brand-new cyberpunk parallax background) ---
const neonParallaxCity = {};
neonParallaxCity.vsSource = quadVS;
neonParallaxCity.fsSource = `
    precision highp float;
    uniform vec2 u_resolution;
    uniform float u_time;

    // Helpers
    float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453); }
    float noise(vec2 p){
        vec2 i = floor(p);
        vec2 f = fract(p);
        f = f*f*(3.0 - 2.0*f);
        float a = hash(i);
        float b = hash(i + vec2(1.0,0.0));
        float c = hash(i + vec2(0.0,1.0));
        float d = hash(i + vec2(1.0,1.0));
        return mix(mix(a,b,f.x), mix(c,d,f.x), f.y);
    }
    float fbm(vec2 p){
        float v = 0.0;
        float a = 0.5;
        mat2 m = mat2(1.6,1.2,-1.2,1.6);
        for(int i=0;i<4;i++){
            v += a * noise(p);
            p = m * p;
            a *= 0.5;
        }
        return v;
    }
    mat2 rot(float a){ return mat2(cos(a), -sin(a), sin(a), cos(a)); }

    // Cyberpunk palette: cyan -> magenta -> violet -> electric blue
    vec3 palette(float t){
        t = clamp(t, 0.0, 1.0);
        vec3 c1 = vec3(0.05, 0.95, 1.00);
        vec3 c2 = vec3(0.95, 0.20, 1.00);
        vec3 c3 = vec3(0.55, 0.20, 0.90);
        vec3 c4 = vec3(0.15, 0.55, 1.00);
        if (t < 0.33) {
            float k = smoothstep(0.0, 0.33, t);
            return mix(c1, c2, k);
        } else if (t < 0.66) {
            float k = smoothstep(0.33, 0.66, t);
            return mix(c2, c3, k);
        } else {
            float k = smoothstep(0.66, 1.0, t);
            return mix(c3, c4, k);
        }
    }

    // Holographic billboard shard SDF (rotated rectangles)
    float sdBox(vec2 p, vec2 b){
        vec2 d = abs(p) - b;
        return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
    }

    // Draw neon grid lines
    float neonGrid(vec2 p, float scale, float thickness){
        vec2 gp = p * scale;
        vec2 g = abs(fract(gp) - 0.5);
        float line = min(g.x, g.y);
        float d = smoothstep(thickness, thickness*0.5, line);
        return d;
    }

    void main(){
        vec2 R = u_resolution;
        vec2 uv = (gl_FragCoord.xy / R);
        vec2 p = uv * 2.0 - 1.0;
        p.x *= R.x / R.y;

        float t = u_time;

        // Layer 1: Distant city glow (parallax slow)
        vec2 l1 = p * 0.6 + vec2(0.05*sin(t*0.07), 0.03*cos(t*0.05));
        float h1 = 0.5 + 0.5 * fbm(l1 * 1.5 + vec2(0.0, t*0.03));
        vec3 col1 = palette(0.25 + 0.35 * h1);
        col1 *= 0.25 + 0.75 * smoothstep(-1.0, 1.2, p.y + 0.15*sin(t*0.2)); // horizon emphasis

        // Layer 2: Neon ground grid (medium parallax, perspective warp)
        vec2 gp = p;
        gp.y += 0.35;                     // move grid down
        float persp = 1.5 / max(0.2, gp.y + 1.6); // fake perspective factor
        vec2 gridUV = vec2(gp.x, gp.y) * vec2(persp*1.1, persp*0.4);
        gridUV.x += t * 0.2;              // scroll forward
        float gridLines = neonGrid(gridUV, 8.0, 0.06);
        vec3 gridCol = mix(vec3(0.0), vec3(0.05, 0.9, 1.0), gridLines);
        // add secondary magenta lines offset
        float gridLines2 = neonGrid(gridUV + vec2(0.23, 0.17), 8.0, 0.06);
        gridCol += vec3(0.9, 0.2, 1.0) * gridLines2 * 0.6;
        // fade with distance and vignette
        float gridFade = smoothstep(-0.2, 0.6, p.y) * (0.9 - 0.7*length(p*vec2(0.9,1.2)));
        gridCol *= gridFade;

        // Layer 3: Floating holographic billboards (faster parallax shards)
        vec3 shardAccum = vec3(0.0);
        float shardAlpha = 0.0;
        vec2 bp = p * 1.4 + vec2(0.2*sin(t*0.41), -0.1*cos(t*0.33));
        bp = rot(0.12*sin(t*0.2)) * bp;
        // sample a small grid of shards, jittered
        for(int j=-1;j<=1;j++){
            for(int i=-1;i<=1;i++){
                vec2 cell = vec2(float(i), float(j));
                vec2 id = floor(bp*3.0) + cell;
                // stable 2D random using two hash() calls
                float hx = hash(id);
                float hy = hash(id + vec2(37.2, 91.7));
                vec2 rnd = vec2(hx, hy);
                vec2 center = (floor(bp*3.0) + cell + 0.5 + (rnd-0.5)*0.35) / 3.0;
                vec2 lp = bp - center;
                float ang = 3.14159 * (rnd.x - 0.5) + 0.3*sin(t*0.7 + dot(id, vec2(0.7,1.3)));
                lp = rot(ang) * lp;
                float sx = mix(0.10, 0.35, rnd.x);
                float sy = mix(0.04, 0.18, rnd.y);
                float d = sdBox(lp, vec2(sx, sy));
                float edge = 1.0 - smoothstep(0.005, 0.02, abs(d));
                float glow = 1.0 - smoothstep(0.09, 0.35, abs(d));
                float hue = fract(0.3 + 0.4*rnd.x + 0.3*sin(t*0.25 + rnd.y*6.0));
                vec3 c = palette(hue);
                vec3 sCol = c * (edge*1.2 + glow*0.6);
                float df = smoothstep(-0.8, 0.4, p.y);
                sCol *= df;
                float w = exp(-6.0*dot(lp, lp));
                shardAccum += sCol * w;
                shardAlpha += (edge + 0.5*glow) * w;
            }
        }
        shardAccum /= max(0.0001, shardAlpha);

        // Layer 4: Volumetric fog bands (fast parallax)
        vec2 fogP = p * 1.8 + vec2(0.35*t, -0.15*t);
        float f1 = fbm(fogP*0.8);
        float f2 = fbm(fogP*1.6 + vec2(3.1, -1.7));
        float fog = smoothstep(0.55, 1.0, f1*0.6 + f2*0.5);
        vec3 fogCol = mix(vec3(0.02,0.03,0.06), vec3(0.05, 0.55, 1.0), 0.6) * fog * 0.35;

        // Compose layers
        vec3 col = vec3(0.01, 0.02, 0.04);
        col = mix(col, col1, 0.85);
        col += gridCol;
        col = mix(col, shardAccum, 0.55);
        col += fogCol;

        // Scanlines and vignette
        float scan = 0.016 * sin(gl_FragCoord.y * 3.14159 + t * 3.2);
        col += vec3(scan) * 0.012;
        float vig = 0.92 - 0.55*dot(p, p);
        col *= clamp(vig, 0.25, 1.0);

        gl_FragColor = vec4(max(col, 0.0), 1.0);
    }
`;
neonParallaxCity.init = () => {
    neonParallaxCity.program = createProgram(gl,
        createShader(gl, gl.VERTEX_SHADER, neonParallaxCity.vsSource),
        createShader(gl, gl.FRAGMENT_SHADER, neonParallaxCity.fsSource)
    );
    gl.useProgram(neonParallaxCity.program);
    neonParallaxCity.positionAttributeLocation = gl.getAttribLocation(neonParallaxCity.program, 'a_position');
    neonParallaxCity.resolutionUniformLocation = gl.getUniformLocation(neonParallaxCity.program, 'u_resolution');
    neonParallaxCity.timeUniformLocation = gl.getUniformLocation(neonParallaxCity.program, 'u_time');
};
neonParallaxCity.draw = (time) => {
    gl.useProgram(neonParallaxCity.program);
    gl.uniform2f(neonParallaxCity.resolutionUniformLocation, canvas.width, canvas.height);
    gl.uniform1f(neonParallaxCity.timeUniformLocation, time / 1000.0);
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.enableVertexAttribArray(neonParallaxCity.positionAttributeLocation);
    gl.vertexAttribPointer(neonParallaxCity.positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
};
neonParallaxCity.name = 'Neon Parallax City';
effects.push(neonParallaxCity);

// Removed per-effect insertion; centralized array defines order

// --- Effect: Chromatic Voronoi Bloom (brand-new volumetric demoscene effect) ---
const chromaticVoronoiBloom = {};
chromaticVoronoiBloom.vsSource = quadVS;
chromaticVoronoiBloom.fsSource = `
    precision highp float;
    uniform vec2 u_resolution;
    uniform float u_time;
    uniform vec2 u_rot; // precomputed cos,sin for a small shared rotation

    // Cheap hash/noise
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

    // Single-cell nearest distance in 2D (3x3 neighborhood), returns squared distance (no sqrt)
    float nearest2_sq(vec2 q){
        vec2 g = floor(q);
        vec2 f = fract(q);
        float dmin = 1e9;
        for(int j=-1;j<=1;j++){
            for(int i=-1;i<=1;i++){
                vec2 o = vec2(float(i), float(j));
                vec2 id = g + o;
                vec2 r = o + vec2(hash(id), hash(id+vec2(7.3,11.1))) - f;
                float d = dot(r,r);
                dmin = min(dmin, d);
            }
        }
        return dmin; // keep squared
    }

    // 3 projected planes, but avoid trig by using shared u_rot for both rotations
    // We rotate p.xy and p.yz with same 2x2 rot matrix to approximate tri-planar symmetry
    float voronoiTri(vec3 p){
        // small scale to keep cells visible
        const float scale = 1.15;
        vec2 cs = u_rot;          // cs.x = cos, cs.y = sin
        mat2 R = mat2(cs.x, -cs.y, cs.y, cs.x);

        vec2 p1 = p.xy * scale;
        vec2 p2 = (R * p.yz) * scale;
        vec2 p3 = (R * p.xz) * scale;

        float d1 = nearest2_sq(p1);
        float d2 = nearest2_sq(p2);
        float d3 = nearest2_sq(p3);

        // use min of squared distances; map to edge metric without sqrt
        float dm = min(d1, min(d2, d3));
        // edge when near sites: dm small -> strong edge. Use smooth window on squared domain.
        float edge = 1.0 - smoothstep(0.018, 0.14, dm);
        return edge;
    }

    // Main marcher: add interest with layered neon edges, soft rim and occasional streaks
    vec3 marchFoam(vec3 ro, vec3 rd, float t){
        float total = 0.0;
        vec3 col = vec3(0.0);
        float glow = 0.0;

        // jitter start
        float seed = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898,78.233)) + t*37.2) * 43758.5453);
        total += seed * 0.04;

        // 36 steps max
        for(int i=0;i<36;i++){
            vec3 pos = ro + rd * total;

            // 2-tap domain warp (very cheap)
            float w1 = noise(pos.xy*0.85 + vec2(0.17*t, -0.13*t));
            float w2 = noise(pos.zy*0.85 + vec2(-0.15*t, 0.19*t));
            pos += 0.40 * vec3(w1 - 0.5, w2 - 0.5, (w1 + w2)*0.18);

            // tri-planar voronoi base edge density
            float edge = voronoiTri(pos);

            // Secondary thinner edge layer for sparkle
            float edgeThin = smoothstep(0.015, 0.06, edge) * (1.0 - smoothstep(0.35, 0.9, edge));

            // Flow phase to animate color and streaks
            float phase = pos.x*0.42 - pos.y*0.33 + pos.z*0.28 + t*0.6;

            // adaptive step: go faster in empty space, slower near edges
            float dStep = mix(0.048, 0.18, 1.0 - edge);
            total += dStep;

            // base palette sweep
            float hue = 0.5 + 0.5 * sin(phase);
            vec3 baseA = vec3(0.05,0.90,1.00);
            vec3 baseB = vec3(0.95,0.20,1.00);
            vec3 base = mix(baseA, baseB, hue);

            // subtle tri-hue shift for richness
            vec3 accent = mix(vec3(1.00,0.75,0.20), vec3(0.20,1.00,0.75), 0.5 + 0.5*sin(phase*0.7));

            // pulse to keep motion lively
            float pulse = 0.65 + 0.35 * sin(0.8*t + pos.x*0.7 + pos.y*0.9 + pos.z*0.5);

            // neon edge with accent mix
            float cEdge = edge * (0.030 + 0.045*pulse);
            vec3 c1 = base * cEdge * 0.65;
            vec3 c2 = accent * edgeThin * 0.25;

            // no flashing/glitch streaks per request
            vec3 contribCol = c1 + c2;
            col += contribCol;
            glow += (cEdge + edgeThin*0.02) * 0.55;

            // Early outs: keep thresholds low for speed
            if (glow > 0.52 || total > 5.2) break;
        }

        vec3 bg = vec3(0.01, 0.015, 0.03);
        col = mix(bg, col, clamp(glow*1.35, 0.0, 1.0));
        col += vec3(glow) * 0.22; // subtle bloom-ish

        return col;
    }

    // Cheap chromatic aberration: slight, time-varying skew for liveliness
    vec3 chromaSkew(vec3 c){
        float k = 0.012 + 0.006*sin(u_time*0.9);
        float k2 = 0.010 + 0.005*cos(u_time*0.7);
        mat3 M = mat3(
            1.00,   k,   0.0,
           -k2,   0.99,  k,
            0.0,  -k2,  1.00
        );
        return clamp(M * c, 0.0, 1.0);
    }

    void main(){
        vec2 R = u_resolution;
        vec2 uv = (gl_FragCoord.xy / R)*2.0 - 1.0;
        uv.x *= R.x / R.y;

        float t = u_time * 0.82;

        // camera
        float r = 3.6;
        vec3 ro = vec3(0.0, 0.35 + 0.2*sin(t*0.38), -r);
        vec3 ta = vec3(0.0, 0.0, 0.0);

        vec3 ww = normalize(ta - ro);
        vec3 uu = normalize(cross(vec3(0.0,1.0,0.0), ww));
        vec3 vv = cross(ww, uu);

        float fov = 1.12;
        vec3 rd = normalize(uu*uv.x + vv*uv.y + ww*fov);

        vec3 col = marchFoam(ro, rd, t);

        // vignette
        float vig = 0.94 - 0.50*dot(uv, uv);
        col *= clamp(vig, 0.30, 1.0);

        // scanline + grain (cheap)
        float scan = 0.008 * sin(gl_FragCoord.y * 3.14159 + t * 3.0);
        col += vec3(scan) * 0.006;
        float grain = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898,78.233)) + t*29.2)*43758.5453) - 0.5;
        col += vec3(grain) * 0.004;

        // single-pass chroma approximation
        col = chromaSkew(col);

        gl_FragColor = vec4(max(col, 0.0), 1.0);
    }
`;
chromaticVoronoiBloom.init = () => {
    chromaticVoronoiBloom.program = createProgram(gl,
        createShader(gl, gl.VERTEX_SHADER, chromaticVoronoiBloom.vsSource),
        createShader(gl, gl.FRAGMENT_SHADER, chromaticVoronoiBloom.fsSource)
    );
    gl.useProgram(chromaticVoronoiBloom.program);
    chromaticVoronoiBloom.positionAttributeLocation = gl.getAttribLocation(chromaticVoronoiBloom.program, 'a_position');
    chromaticVoronoiBloom.resolutionUniformLocation = gl.getUniformLocation(chromaticVoronoiBloom.program, 'u_resolution');
    chromaticVoronoiBloom.timeUniformLocation = gl.getUniformLocation(chromaticVoronoiBloom.program, 'u_time');
    chromaticVoronoiBloom.rotUniformLocation = gl.getUniformLocation(chromaticVoronoiBloom.program, 'u_rot');
};
chromaticVoronoiBloom.draw = (time) => {
    gl.useProgram(chromaticVoronoiBloom.program);
    gl.uniform2f(chromaticVoronoiBloom.resolutionUniformLocation, canvas.width, canvas.height);
    const t = time / 1000.0;
    gl.uniform1f(chromaticVoronoiBloom.timeUniformLocation, t);
    // precompute a tiny shared rotation angle once per frame for shader
    const ang = 0.18 * Math.sin(t*0.23) + 0.12 * Math.sin(t*0.11);
    gl.uniform2f(chromaticVoronoiBloom.rotUniformLocation, Math.cos(ang), Math.sin(ang));

    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.enableVertexAttribArray(chromaticVoronoiBloom.positionAttributeLocation);
    gl.vertexAttribPointer(chromaticVoronoiBloom.positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
};
chromaticVoronoiBloom.name = 'Chromatic Voronoi Bloom';
effects.push(chromaticVoronoiBloom);

// Removed per-effect ensure; centralized array already lists all
// --- Main Animation Loop ---
function animateEffects(currentTime) {
    requestAnimationFrame(animateEffects);

    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // FPS update
    const now = currentTime;
    const dt = now - lastFrameTime;
    lastFrameTime = now;
    fpsFrames++;
    fpsAccumTime += dt;
    if (fpsAccumTime >= 500) {
        fps = 1000.0 * fpsFrames / fpsAccumTime;
        fpsFrames = 0;
        fpsAccumTime = 0;
        updateEffectIndicator();
    }

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
    if (gl) gl.viewport(0, 0, canvas.width, canvas.height);
    // Also update projection-dependent effects state lazily on next draw
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
    if (!DEBUG_INFO) {
        effectIndicator.style.display = 'none';
        return;
    }
    effectIndicator.style.display = 'block';
    const name = (effects[currentEffectIndex] && effects[currentEffectIndex].name) ? effects[currentEffectIndex].name : `Effect ${currentEffectIndex + 1}`;
    const res = `${canvas.width}x${canvas.height}`;
    effectIndicator.textContent = `Effect: ${name} (${currentEffectIndex + 1}/${effects.length || 0})  |  FPS: ${fps.toFixed(0)}  |  ${res}`;
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

