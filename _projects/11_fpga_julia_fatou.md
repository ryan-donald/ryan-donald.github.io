---
layout: page
title: Animated Julia and Fatou Sets on an FPGA
description: A VHDL VGA driver and fractal generator targeting an Intel DE10-Lite
img: assets/img/thumb_julia.jpg
importance: 1
category: Hardware
---

Within my Advanced Digital Systems class I implemented a VGA driver, alongside math to calculate and create images displaying Julia and Fatou sets on a monitor in VHDL, targeting an Intel DE-10 Lite board. In implementing this, a VGA driver was first developed, which implemented logic to drive the correct signals, including the displayed data, the front porch, back porch, and sync signals, alongside the VGA clock. This driver was able to drive a monitor at 1920x1080 @ 60Hz, 640x480 @ 60Hz, and 800x600 @ 60Hz. Additionally, we calculated the conversions from a windowed view of the Julia Fatou set, to specific pixels in the monitor. A video of the monitor output can be found:

<video src="{{ site.baseurl }}/assets/img/JuliaFatouVideo.mp4" width="100%" controls loop muted playsinline></video>
