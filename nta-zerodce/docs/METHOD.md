# Method specification

## Notation and provenance

Baseline architecture/curve motivation: [R1](REFERENCES.md#r1). Reference-profile
reductions/weights: [R2](REFERENCES.md#r2). All extension formulas below are
project proposals, not published Zero-DCE results. Numerical ranges and initial
scales are analytical expectations or diagnostic ranges, never measured results.
Tunable constants also live explicitly in baseline/full YAML; changing them changes the
protocol. [INTERFACES](INTERFACES.md) fixes device/layout and flow conventions.

```text
x,z,o: float32 RGB [B,3,H,W], values [0,1]; z=enhanced_raw, o=enhanced
A: [B,24,H,W], [-1,1]; channels (i*3+c) for i=0..7, c=R,G,B
sigma: [B,1,H,W], [0,.2]; sigma=0 if noise_aware=false
Y(x) = (x_R+x_G+x_B)/3             # compatibility intensity, not linear luminance
P_k: non-overlapping mean pool k*k, stride k, no padding
G: detached local gain [B,1,H,W], [1,8]
eps = 1e-6
mean = arithmetic mean over stated valid indices, including batch
masked_mean_C(v,m) = sum(m*v)/(C*sum(m)+eps)
```

For losses, an empty mask returns differentiable zero tied to the output tensor;
for reporting metrics it yields NaN and zero coverage. Geometry uses RGB display
values; the synthetic sensor model is only an approximation here. Require H,W
divisible by 16 in training; arbitrary inference sizes need H,W>=16. Pooling drops
incomplete right/bottom regions at evaluation; never silently pad with dark pixels.

## Baseline network and curve

The planned baseline uses seven 3x3 stride-one convolutions, same-size zero padding,
32 hidden channels, ReLU hidden activations, symmetric concatenations, and a
24-channel tanh output. No normalization or downsampling in the estimator.
These architectural choices follow [R1](REFERENCES.md#r1); the explicit connection
contract is written below so future implementation cannot guess the skips.

```text
h1 = relu(conv(in_channels -> 32, x_or_concat))
h2 = relu(conv(32 -> 32, h1))
h3 = relu(conv(32 -> 32, h2))
h4 = relu(conv(32 -> 32, h3))
h5 = relu(conv(64 -> 32, concat(h3,h4)))
h6 = relu(conv(64 -> 32, concat(h2,h5)))
A  = tanh(conv(64 -> 24, concat(h1,h6)))
x_0 = x
x_(i+1) = x_i + A_i*x_i*(1-x_i), i=0..7
z = x_8
```

The plus-sign curve is this repository's canonical convention. Some reference
implementations express it with the opposite parameter sign; do not directly
load their weights. For fixed A in [-1,1], endpoints are fixed and the map is
bounded and monotone; a spatially/input-dependent predicted A does not imply
the entire learned model is globally monotone. A=0 is identity.

```text
d LE(x,A)/dx = 1 + A*(1-2*x) in [0,2] for fixed A
LE(.25,.5)=.34375; LE(.25,-.5)=.15625
```

The light variant substitutes each hidden 3x3 convolution with depthwise 3x3
then pointwise 1x1, uses the same skips/width, and retains a regular final 3x3
tanh head. NoiseBranch uses RGB->16->16->1 3x3 layers, hidden ReLU, bounded
sigmoid output. ResidualDenoiser uses concatenated [z,sigma*G] (four channels),
two depthwise/pointwise 16-channel blocks and regular RGB residual head. These
auxiliary details are untuned design choices, not claims from [R3](REFERENCES.md#r3).

## Baseline objectives and reduction profile

```text
u=P_4(Y(x)); v=P_4(Y(o)); K=B*pooled_H*pooled_W
L_spa = (1/K)*sum_(i,j in valid 4-neighbours(i)) (|v_i-v_j|-|u_i-u_j|)^2
q=P_16(Y(o))
L_exp_paper = mean |q-.6|
L_exp = mean (q-.6)^2                 # selected reference-code profile
mu_c = mean_(H,W) o_c
d_cd = mu_c-mu_d
L_col_paper = mean_B (d_RG^2+d_RB^2+d_GB^2)
L_col = mean_B sqrt(d_RG^4+d_RB^4+d_GB^4)  # selected reference-code profile
L_tv_paper = mean_(i,c) (|dx A|+|dy A|)^2  # paper notation, boundary alignment needed
L_tv = 2*(mean(dx(A)^2) + mean(dy(A)^2))   # independent squared-difference means
L_base = 1*L_spa + 10*L_exp + 5*L_col + 200*L_tv
```

Both directional TV terms average squared finite differences independently:

```text
L_tv = 2*(sum(dx(A)^2)/numel(dx(A)) + sum(dy(A)^2)/numel(dy(A)))
```

Paper formulas and reference-profile reductions differ [R1](REFERENCES.md#r1),
[R2](REFERENCES.md#r2); M1 must report this profile and not claim exact paper or
code parity. Spa uses only valid neighbor pairs (no padded boundary penalties).
The colour square-root norm is computed with a zero-safe derivative at all-zero
differences in future implementation, preserving exact zero loss there.

## Extension 1: noise estimation and denoise

Sample per-image coefficients uniformly within YAML bounds; one coefficient
pair shared by channels, independent standard normals per pixel/channel.

```text
a ~ Uniform(0,.01), b ~ Uniform(0,.02), a,b: [B,1,1,1]
epsilon ~ N(0,1), [B,3,H,W]
var_c = a*x_c + b^2
n_c = sqrt(var_c)*epsilon_c
x_noisy = clamp(x+n,0,1)
sigma_target = sqrt(a*Y(x)+b^2)        # [B,1,H,W], before clipping
sigma_hat = .2*sigmoid(noise_head(x_noisy))
input_DCE = concat(x_noisy,sigma_hat)  # [B,4,H,W]
G = stopgrad(clamp(avgpool_5(Y(z))/(avgpool_5(Y(x_noisy))+eps),1,8))
r = .1*tanh(denoise_head(concat(z,sigma_hat*G)))
o = clamp(z-r,0,1)                    # r RGB [B,3,H,W]
L_noise = mean |sigma_hat-sigma_target|
t = stopgrad(E(x).enhanced)           # same model, no_grad, same augmentation
t_raw = stopgrad(E(x).enhanced_raw)
L_invariance = mean |E(x_noisy).enhanced-t|
L_fidelity = mean |E(x_noisy).enhanced-t_raw|  # only when denoise enabled
```

avgpool_5 uses stride 1 and reflection padding. Noise synthesis is a Gaussian
approximation to heteroscedastic Poisson-Gaussian noise with the requested
variance; it does not sample exact Poisson counts. Clipping changes observed
variance and introduces bias near boundaries; supervision uses the known
pre-clipping target. SICE inputs already contain noise, so added-noise sigma
is not a ground-truth total-camera-noise label. The teacher is imperfect;
fidelity/invariance can conflict with intended brightness changes. Never use LOL
high images in this unpaired training objective. Gain is a bounded heuristic,
not the derivative of the curve nor an exact propagated-noise model.

## Extension 2: edge preservation

```text
s = GaussianBlur(Y(x), sigma=.8, kernel=5, reflect padding)
Sx = [[-1,0,1],[-2,0,2],[-1,0,1]]/8; Sy=transpose(Sx)
D(v) = sqrt((Sx*v)^2+(Sy*v)^2+eps^2)-eps  # [B,1,H,W]
d_in = stopgrad(D(s)); d_out=D(Y(o))
w_edge = stopgrad(clamp((d_in-2*sigma)/(d_in+eps),0,1))
L_edge = masked_mean_1(|d_out-stopgrad(G*d_in)|,w_edge)
```

The unblurred sigma threshold is deliberately conservative for smoothed gradients;
it is not a statistically calibrated Sobel noise variance. Flat/noisy regions get
little weight. Detach input sigma/gain/weights, retain output gradient. Edge-only
ablation uses sigma=0. G is pooled gain as above, independent of denoise flag.

## Extension 3: local chromaticity

```text
c(v)=v/(sum_RGB(v)+eps)                # [B,3,H,W], each component [0,1]
c_ref=stopgrad(avgpool_5(c(x)))         # stride1, reflect padding
w_chroma=stopgrad(smoothstep(.05,.2,Y(x)-2*sigma))
L_chroma=masked_mean_3(|c(o)-c_ref|,w_chroma)
```

Local averaging occurs after normalization, not normalization of averaged RGB.
Black pixels remain near zero and get unreliable weights. Input colour is a
constraint, not proof of accurate colour: real colour casts may be preserved.

## Extension 4: adaptive exposure and highlights

```text
smoothstep(l,h,v): t=clamp((v-l)/(h-l),0,1); return t^2*(3-2*t)
Yp=P_16(Y(x)); sigma_p=P_16(stopgrad(sigma))
T_p=stopgrad(.6+(Yp-.6)*smoothstep(.55,.85,Yp)
                 -.15*clamp(sigma_p/.05,0,1))
L_adaptive_exp=mean (P_16(Y(o))-T_p)^2   # T_p [B,1,H//16,W//16]
m_hi=stopgrad(1[Y(x)>.8])
L_highlight=masked_mean_1(relu(Y(o)-Y(x))^2,m_hi)
```

Targets lie in [.45,1] for the specified constants. Adaptive replaces baseline
exp at the same weight, rather than adding a second exposure objective. Highlight
protection penalizes further brightening of existing sources, not all high output
pixels; it cannot restore input clipping. With sigma=0 the noise penalty vanishes.

## Extension 5: temporal training and inference

Flow convention follows a source-to-target displacement, estimated by DIS
[R6](REFERENCES.md#r6); bidirectional consistency is a project mask choice.
The general temporal motivation is related to [R9](REFERENCES.md#r9), not a
reproduction of its recurrent method. Our inference requires flow.

```text
f=f(frame_(t-1),frame_t); b=f(frame_t,frame_(t-1))
warp(v,b)(p)=bilinear_sample(v,p+b(p))
r_fb = b + warp(f,b)
m_fb = 1[sum_xy(r_fb^2) <= .01*(sum_xy(b^2)+sum_xy(warp(f,b)^2))+.5]
m_photo = 1[|Y(frame_t)-Y(warp(frame_(t-1),b))| < .1]
m = stopgrad(m_fb*m_photo*in_bounds(b))  # [B,1,H,W]
L_temporal=masked_mean_3(|o_t-warp(o_(t-1),b)|,m)
```

Flow/masks use unenhanced frames and remain detached. Both enhanced outputs
retain gradients. Exposure flicker can invalidate the photometric test; report
coverage and use known geometric validity for synthetic pairs. Analytic pairs:

```text
image2(q)=image1(T^-1(q))               # affine transform in pixel coordinates
f(p)=T(p)-p; b(q)=T^-1(q)-q
image2 <- clamp(flicker*image2+independent_noise,0,1), flicker ~ U(.9,1.1)
rotation ~ U(-3,3) degrees; translation_xy ~ U(-8,8) pixels
scale ~ U(.98,1.02); geometric mask excludes transformed border samples
```

Synthetic mask excludes invalid transformed borders and FB failures, but bypasses
the photometric threshold because controlled flicker is an intended nuisance.
Use both directions analytically; negating forward flow works only for a constant
translation. Normalize a grid as below; singleton sizes are rejected.

```text
grid_x=2*(p_x+b_x)/(W-1)-1; grid_y=2*(p_y+b_y)/(H-1)-1
grid_sample(...,mode=bilinear,padding=zeros,align_corners=True)
```

At inference, first estimate current maps; compare 32-bin mean-RGB histograms
normalized to sum one. Cut distance uses half L1 histogram distance in [0,1].

```text
cut = .5*sum(|hist32(Y(current))-hist32(Y(previous))|) > .3
if first_frame or cut or changed_shape: reset; A_bar_t=A_t
else: A_bar_t=(1-.7*m)*A_t + .7*m*warp(A_bar_(t-1),b)
z_t=apply_curves(current,A_bar_t)
o_t=optional_denoise(z_t,sigma_t,gain_t)
o_final=(1-beta*m)*o_t+beta*m*warp(o_final_(t-1),b); beta=0 by default
```

Same mask broadcasts over 24 curve channels. Convex blends preserve curve range.
Store smoothed maps, not unsmoothed maps. Scene-cut false negatives cause history
leakage; false positives remove smoothing. Output blending is off by default to
limit ghosting; the heuristic cut threshold needs real-video validation.

## Total objective and default weights

| Key | Weight | Activation | Why start here |
| --- | --- | --- | --- |
| spa | 1 | always | reference-profile anchor |
| exp | 10 | baseline or adaptive replacement | reference-profile anchor |
| col | 5 | always | reference-profile anchor |
| tv | 200 | always | reference-profile anchor; small squared map differences |
| edge | 20 | edge | compensate small normalized gradient residuals |
| chroma | 20 | chroma | regularize local RGB proportions |
| highlight | 20 | adaptive_exp | focus a sparse masked term |
| noise | 20 | noise_aware | sigma errors are in small RGB units |
| invariance | 10 | noise_aware | consistency without real high-light GT |
| fidelity | 1 | denoise | weak synthetic teacher anchor |
| temporal | 10 | temporal | normalized RGB L1 consistency |

```text
L_total=L_spa+10*(L_adaptive_exp if adaptive_exp else L_exp)+5*L_col+200*L_tv
        +20*edge_flag*L_edge+20*chroma_flag*L_chroma
        +20*adaptive_exp_flag*L_highlight
        +20*noise_flag*L_noise+10*noise_flag*L_invariance
        +denoise_flag*L_fidelity+10*temporal_flag*L_temporal
```

These are untuned starting values supplied by the project brief. Reduction choices,
mask coverage and noise distribution change relative gradient scales. Log raw and
weighted losses and gradient norms before tuning on training validation only.

## Initial-loss scale checks (not results)

Use an identity-output diagnostic, zero/constant A, and constant gray input .1.
Random initialization need not be identity, especially bounded sigmoid sigma.

| Term | Analytical diagnostic / plausible scale | Interpretation |
| --- | --- | --- |
| spa | exactly 0 at identity; often 0..1e-2 for small changes | squared regional contrast |
| exp | (.1-.6)^2=.25; weighted 2.5 | dominates dark flat identity |
| col | exactly 0 for equal RGB means; about 1e-3..1e-1 if casts present | square-root fourth powers |
| tv | exactly 0 for constant A; no guaranteed random-init scale | high weight can dominate rough maps |
| noise | .1 predicted vs .02 target => .08, weighted 1.6 | sigmoid near zero logits predicts .1 |
| invariance/fidelity | identity Gaussian abs noise about sqrt(2/pi)*sigma per RGB channel away from clipping | order 1e-2..1e-1 for chosen corruption |
| edge | 0 for flat identity or empty mask; otherwise no data-independent scale | inspect gain and edge weights |
| chroma | 0 for reliable uniform gray; may be 1e-3..1e-1 with changes | local averaging can make identity nonzero on texture |
| adaptive exp | (.1-.45)^2=.1225 at saturated sigma penalty | weighted 1.225 |
| highlight | 0 for dark gray; .1 brightening gives .01 on covered pixels | sparse mask uses covered-pixel normalization |
| temporal | 0 for correctly warped unchanged pair; residual noise/flicker order 1e-2..1e-1 | flow errors can dominate |

Ranges are sanity guides, not measured expectations from a network. At M1/M2
record actual loss histograms and mask coverage on fixed batches. Do not tune
against held-out LOL evaluation or invent results from these examples.

Adam uses beta1=.9, beta2=.999, eps=1e-8 explicitly in YAML; the optimizer
is Adam with coupled weight decay, not AdamW. Gain pooling width and cut histogram
size are model.gain_window=5 and temporal.histogram_bins=32.
