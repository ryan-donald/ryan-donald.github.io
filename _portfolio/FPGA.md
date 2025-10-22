---
title: "Animated Julia and Fatou set on an FPGA"
excerpt: "Implementation of a VGA driver, and an animated Julia and Fatou set"
collection: portfolio
---
Within my Advanced Digital Systems class I implemented a VGA driver, alongside math to calculate and create images displaying Julia and Fatou sets on a monitor in VHDL, targeting an Intel DE-10 Lite board. In implementing this, a VGA driver was first developed, which implemented logic to drive the correct signals, including the displayed data, the front porch, back porch, and sync signals, alongside the VGA clock. This driver was able to drive a monitor at 1920x1080 @ 60Hz, 640x480 @ 60Hz, and 800x600 @ 60Hz. Additionally, we calculated the conversions from a windowed view of the Julia Fatou set, to specific pixels in the monitor. A video of the monitor output can be found:

https://github.com/ryan-donald/ryan-donald.github.io/raw/refs/heads/master/images/JuliaFatouVideo.mp4
