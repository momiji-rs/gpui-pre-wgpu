# gpui-pre-wgpu, patched

[gpui-pre-wgpu](https://crates.io/crates/gpui-pre-wgpu) 0.3.7, the crates.io
snapshot of Zed's `gpui_wgpu` crate (zed-industries/zed@1a28cff), with one
fix that has not been released yet:

- **Skip the GL backend when Vulkan works**, backported from
  [zed-industries/zed#63313](https://github.com/zed-industries/zed/pull/63313)
  by João Lucas. With Vulkan alone the Mesa GL stack is never loaded:
  measured in [Kissa](https://github.com/momiji-rs/kissa) on a Ryzen 7 8745HS (Radeon 780M, RADV)
  under Hyprland (2026-10-04), resident memory fell from 96 MiB to 75 MiB.
  With no Vulkan driver it still falls back to GL and draws.

The first commit is the crate exactly as published; the fix is the second.
Use it with a pinned revision:

```toml
[patch.crates-io]
gpui-pre-wgpu = { git = "https://github.com/momiji-rs/gpui-pre-wgpu", rev = "…" }
```

Retire this fork once a gpui-pre-wgpu release includes the fix.
