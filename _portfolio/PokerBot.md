---
title: "PokerBot"
excerpt: "A Texas Hold'em playing Robot"
collection: portfolio
---

This was a robot developed to play Texas Hold-em against real players within a group of three. This was developed as the final project for Advanced Real-World Robotics at UMass Lowell under Professor Robinette. This robot consisted of a vision model, a reinforcement learning agent capable of playing Texas Hold-em, and a table-top robot arm with a suction-cup manipulator. For the vision model, I created a dataset of images for each card and then trained the YOLO-v8 model on this dataset. This model was able to distinguish the card value and suit, which was then input into the reinforcement learning agent. The reinforcement learning agent was a Deep Q-Network, trained utilizing an open-source texas-holdem simulator. Lastly, the table-top robot arm was able to detect cards within a space designated for it's own cards, pick those cards up face-down, and move them over a camera embedded in the table. A second camera was mounted over the table, and was able to detect the other player's cards.
