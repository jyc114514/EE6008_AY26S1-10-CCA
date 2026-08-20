# EE6008 AY26S1-10-CCA scope v0.3

This document separates evidence-backed project constraints from working hypotheses. It is a planning baseline, not a claim that the team has already approved every item.

## Confirmed

- The course project is AY26S1-10-CCA, titled Joint-Embedding Predictive Modelling for Humanoid Everyday Action Anticipation.
- The project should start from an existing V-JEPA-class implementation rather than train a JEPA from scratch.
- The first milestone should be a small, inspectable dataset/metadata route and a conventional baseline before adding a representation or prediction modification.
- GitHub is for code, configs, metadata, notes, and small final summaries; raw data, checkpoints, features, outputs, and logs belong on the GPU Server.
- A real result must be traceable to a dataset revision, split, config, Git commit, run ID, seed, model identifier, owner, date, and metrics.

## Tentative

- Start with one Humanoid Everyday robot subset (G1 or H1), selected after checking task completeness and action dimensions.
- Compare a conventional video/sequence baseline with a frozen V-JEPA/V-JEPA 2 representation and a small downstream head.
- Add action anticipation or a related future-action task only after current-action recognition and label alignment are working.
- Candidate lightweight extensions include encoder/predictor comparison, temporal pooling choices, or RGB/state fusion. These are hypotheses, not established innovations.

## Open questions

- Which robot subset and task family will the team and supervisor approve?
- What exact dataset revision, license constraints, and episode-level split will be used?
- Which V-JEPA 2/2.1 model and official input contract can be profiled on the assigned GPU?
- What is the primary metric and what constitutes a fair baseline?
- Which four GitHub usernames should receive Write collaborator invitations?

## Out of scope for the initial scaffold

- Training a JEPA from scratch.
- Publishing raw Humanoid Everyday data, videos, checkpoints, extracted tensors, or full training outputs.
- Claiming hardware deployment, benchmark performance, or a final task definition before verification.
- Enterprise infrastructure such as Kubernetes, Docker Swarm, or a complex CI/CD platform.
