/*
 * model.js — Pure JavaScript inference for a GPT-style transformer.
 *
 * This is NOT a copy of the PyTorch model — it's a from-scratch re-implementation
 * of the exact same math (embeddings, causal self-attention, feedforward, layernorm)
 * that reads the weights exported by train_and_export.py and reproduces the same
 * forward pass, entirely client-side. No server, no Python required at runtime.
 */

let META = null;
let WEIGHTS = null; // ArrayBuffer

async function loadModel() {
  const [metaRes, weightsRes] = await Promise.all([
    fetch("meta.json"),
    fetch("weights.bin"),
  ]);
  META = await metaRes.json();
  WEIGHTS = await weightsRes.arrayBuffer();
  return META;
}

// Slice out a named tensor as a Float32Array, given its shape from meta.json.
function getTensor(name) {
  const info = META.tensors[name];
  if (!info) throw new Error(`Tensor not found: ${name}`);
  return new Float32Array(WEIGHTS, info.offset, info.length);
}

// --- Basic math helpers -----------------------------------------------------

// PyTorch nn.Linear weight is stored as [outFeatures, inFeatures] (row-major).
// This computes y = W x + b for a single vector x.
function linear(x, W, b, inFeatures, outFeatures) {
  const y = new Float32Array(outFeatures);
  for (let o = 0; o < outFeatures; o++) {
    let sum = b ? b[o] : 0;
    const rowStart = o * inFeatures;
    for (let i = 0; i < inFeatures; i++) {
      sum += W[rowStart + i] * x[i];
    }
    y[o] = sum;
  }
  return y;
}

function linearSeq(xs, W, b, inFeatures, outFeatures) {
  return xs.map((x) => linear(x, W, b, inFeatures, outFeatures));
}

function layerNorm(x, weight, bias, eps = 1e-5) {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let variance = 0;
  for (let i = 0; i < n; i++) variance += (x[i] - mean) ** 2;
  variance /= n;
  const invStd = 1 / Math.sqrt(variance + eps);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = (x[i] - mean) * invStd * weight[i] + bias[i];
  }
  return out;
}

function layerNormSeq(xs, weight, bias) {
  return xs.map((x) => layerNorm(x, weight, bias));
}

function softmax(scores) {
  const max = Math.max(...scores);
  const exps = scores.map((s) => Math.exp(s - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

// tanh-based GELU approximation (very close to PyTorch's exact GELU for our purposes)
function gelu(x) {
  const out = new Float32Array(x.length);
  const c = Math.sqrt(2 / Math.PI);
  for (let i = 0; i < x.length; i++) {
    const v = x[i];
    out[i] = 0.5 * v * (1 + Math.tanh(c * (v + 0.044715 * v ** 3)));
  }
  return out;
}

function addVec(a, b) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i];
  return out;
}

// --- Transformer forward pass -----------------------------------------------

function attentionHead(xs, layerIdx, headIdx, headSize) {
  const T = xs.length;
  const prefix = `blocks.${layerIdx}.sa.heads.${headIdx}`;
  const Wk = getTensor(`${prefix}.key.weight`);
  const Wq = getTensor(`${prefix}.query.weight`);
  const Wv = getTensor(`${prefix}.value.weight`);
  const nEmbed = META.n_embed;

  const K = xs.map((x) => linear(x, Wk, null, nEmbed, headSize));
  const Q = xs.map((x) => linear(x, Wq, null, nEmbed, headSize));
  const V = xs.map((x) => linear(x, Wv, null, nEmbed, headSize));

  const scale = 1 / Math.sqrt(headSize);
  const out = [];
  for (let t = 0; t < T; t++) {
    // causal: token t can only attend to tokens 0..t
    const scores = [];
    for (let j = 0; j <= t; j++) {
      let dot = 0;
      for (let d = 0; d < headSize; d++) dot += Q[t][d] * K[j][d];
      scores.push(dot * scale);
    }
    const weights = softmax(scores);
    const outVec = new Float32Array(headSize);
    for (let j = 0; j <= t; j++) {
      for (let d = 0; d < headSize; d++) outVec[d] += weights[j] * V[j][d];
    }
    out.push(outVec);
  }
  return out;
}

function multiHeadAttention(xs, layerIdx) {
  const nHead = META.n_head;
  const nEmbed = META.n_embed;
  const headSize = nEmbed / nHead;
  const T = xs.length;

  const headOutputs = [];
  for (let h = 0; h < nHead; h++) {
    headOutputs.push(attentionHead(xs, layerIdx, h, headSize));
  }

  // concat heads along feature dim
  const concat = [];
  for (let t = 0; t < T; t++) {
    const vec = new Float32Array(nEmbed);
    for (let h = 0; h < nHead; h++) {
      vec.set(headOutputs[h][t], h * headSize);
    }
    concat.push(vec);
  }

  const Wproj = getTensor(`blocks.${layerIdx}.sa.proj.weight`);
  const bproj = getTensor(`blocks.${layerIdx}.sa.proj.bias`);
  return linearSeq(concat, Wproj, bproj, nEmbed, nEmbed);
}

function feedForward(xs, layerIdx) {
  const nEmbed = META.n_embed;
  const hidden = 4 * nEmbed;
  const W1 = getTensor(`blocks.${layerIdx}.ffwd.net.0.weight`);
  const b1 = getTensor(`blocks.${layerIdx}.ffwd.net.0.bias`);
  const W2 = getTensor(`blocks.${layerIdx}.ffwd.net.2.weight`);
  const b2 = getTensor(`blocks.${layerIdx}.ffwd.net.2.bias`);

  return xs.map((x) => {
    const h = linear(x, W1, b1, nEmbed, hidden);
    const activated = gelu(h);
    return linear(activated, W2, b2, hidden, nEmbed);
  });
}

function transformerBlock(xs, layerIdx) {
  const ln1w = getTensor(`blocks.${layerIdx}.ln1.weight`);
  const ln1b = getTensor(`blocks.${layerIdx}.ln1.bias`);
  const ln2w = getTensor(`blocks.${layerIdx}.ln2.weight`);
  const ln2b = getTensor(`blocks.${layerIdx}.ln2.bias`);

  const normed1 = layerNormSeq(xs, ln1w, ln1b);
  const attnOut = multiHeadAttention(normed1, layerIdx);
  const xs2 = xs.map((x, t) => addVec(x, attnOut[t]));

  const normed2 = layerNormSeq(xs2, ln2w, ln2b);
  const ffOut = feedForward(normed2, layerIdx);
  const xs3 = xs2.map((x, t) => addVec(x, ffOut[t]));

  return xs3;
}

// Full forward pass. tokenIds: array of ints (already truncated to <= block_size).
// Returns logits (Float32Array of vocab_size) for the NEXT token after the sequence.
function forward(tokenIds) {
  const nEmbed = META.n_embed;
  const tokEmbW = getTensor("token_embedding.weight");
  const posEmbW = getTensor("position_embedding.weight");
  const vocabSize = META.vocab_size;

  let xs = tokenIds.map((id, t) => {
    const tok = tokEmbW.subarray(id * nEmbed, id * nEmbed + nEmbed);
    const pos = posEmbW.subarray(t * nEmbed, t * nEmbed + nEmbed);
    return addVec(tok, pos);
  });

  for (let l = 0; l < META.n_layer; l++) {
    xs = transformerBlock(xs, l);
  }

  const lnfW = getTensor("ln_f.weight");
  const lnfB = getTensor("ln_f.bias");
  xs = layerNormSeq(xs, lnfW, lnfB);

  const lastHidden = xs[xs.length - 1];
  const headW = getTensor("lm_head.weight");
  const headB = getTensor("lm_head.bias");
  return linear(lastHidden, headW, headB, nEmbed, vocabSize);
}

// Sample one token id from logits using temperature-scaled softmax.
function sampleFromLogits(logits, temperature = 0.8) {
  const scaled = Array.from(logits).map((l) => l / temperature);
  const probs = softmax(scaled);
  const r = Math.random();
  let cumulative = 0;
  for (let i = 0; i < probs.length; i++) {
    cumulative += probs[i];
    if (r < cumulative) return i;
  }
  return probs.length - 1;
}

function encode(str) {
  return Array.from(str).map((ch) => {
    const id = META.stoi[ch];
    if (id === undefined) throw new Error(`Character not in vocabulary: "${ch}"`);
    return id;
  });
}

function decode(ids) {
  return ids.map((id) => META.itos[String(id)]).join("");
}

// Generate `maxNewTokens` characters continuing from `prompt`.
// onToken(char) is called after each new character, for streaming display.
async function generate(prompt, maxNewTokens, temperature, onToken) {
  let ids = encode(prompt);
  const blockSize = META.block_size;

  for (let i = 0; i < maxNewTokens; i++) {
    const context = ids.slice(Math.max(0, ids.length - blockSize));
    const logits = forward(context);
    const nextId = sampleFromLogits(logits, temperature);
    ids.push(nextId);
    if (onToken) onToken(decode([nextId]));
    // yield to the browser so the UI stays responsive and text streams in
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  return decode(ids);
}
