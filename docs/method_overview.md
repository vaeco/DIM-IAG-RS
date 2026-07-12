# Method Overview

## 1. Shared DIM-WRR Backbone

Given a low-light image `X`, the DIM-WRR backbone predicts a base enhanced image:

```text
Y_base = F_DIM-WRR(X)
R = Y_base - X
```

The conservative and brightness-oriented checkpoints share the same DIM-WRR backbone. Their primary difference is the learned behavior of the illumination-aware gate.

## 2. Conservative Gate Expert

The conservative expert is intended to preserve task-relevant image statistics and avoid unnecessary enhancement in moderately illuminated or exposure-sensitive inputs.

Its risk-aware suppression is:

```text
S_c = clamp(
    1 - 0.20*N_c - 0.35*O_c + 0.10*U_c,
    0.55,
    1.00
)

E_c = G_c * S_c
```

where `G_c` is the conservative gate, `N_c` is the noise estimate, `O_c` is the over-exposure risk, and `U_c` is the under-exposure map.

## 3. Brightness-Oriented Gate Expert

The brightness-oriented expert is retrained on paired low-light and normal-light images while the DIM-WRR backbone is frozen by default.

Its risk-aware control is:

```text
S_b = clamp(
    1 - 0.08*N_b - 0.22*O_b + 0.30*U_b,
    0.65,
    1.20
)

E_b = G_b * S_b
```

The larger under-exposure coefficient and upper bound allow stronger residual restoration in severely dark regions.

## 4. Continuous Dual-Gate Router

The router does not use dataset identities. It estimates image-level under-exposure severity from the input:

```text
u = mean(U_router)
severity = sigmoid((u - 0.81) / 0.02)
```

Exposure and noise risks form a safety factor:

```text
risk_safety = clamp(
    1 - 1.50*over_mean - 0.50*relu(noise_mean - 0.55),
    0.50,
    1.00
)
```

The final mixing coefficient is:

```text
rho = clamp(
    0.95 * severity * risk_safety,
    0.00,
    0.95
)
```

The final effective gate is:

```text
E = (1 - rho)*E_c + rho*E_b
```

The enhanced image is:

```text
Y = clamp(X + E*(Y_base - X), 0, 1)
```

Thus, moderately illuminated images remain close to the conservative expert, while severely under-exposed images receive a larger contribution from the brightness-oriented expert.

## 5. Current Validation Behavior

With the final candidate parameters:

```text
router_center = 0.81
router_width  = 0.02
rho_max       = 0.95
```

20-image diagnostic runs showed the intended routing behavior:

| Input domain | Mean rho | Mean final effective gate |
|---|---:|---:|
| CODaN diagnostic subset | 0.0727 | 0.4759 |
| LOD/LIS diagnostic subset | 0.7103 | 0.8234 |

These values are diagnostic statistics rather than downstream task metrics.
