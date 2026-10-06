# CB-ELE: related work and competitors

Scope: leader election or leader–follower coordination in swarms with **no inter-agent
communication** and **no GPS/GNSS**. The search was run on 2026-10-06. Each entry was checked
against its arXiv abstract page or an independent index or publisher listing. DOIs are
given where they could be confirmed; otherwise a stable URL is given.

## 1. Leader selection without explicit communication (implicit, motion-based or visual)

1. **Ahmad, Bonilla Licea, Silano, Báča, Saska.** PACNav: a collective navigation approach for UAV swarms deprived of communication and external localization. *Bioinspiration & Biomimetics* 17(6), 2022. DOI 10.1088/1748-3190/ac98e6 · arXiv:2302.05258.
   UAVs infer whom to follow from neighbours' path persistence and similarity, using onboard relative sensing only. This is the closest UAV analogue, but it never elects a single leader and has no re-election.
2. **Choi, Ko, Cho, Kim, Kim, Seo, Oh.** Communication-Free Collective Navigation for a Swarm of UAVs via LiDAR-Based Deep Reinforcement Learning. arXiv:2601.13657, 2026.
   Implicit leader–follower with a deep-RL follower policy on LiDAR. The leader is assigned beforehand, not elected.
3. **Berlinger, Gauci, Nagpal.** Implicit coordination for 3D underwater collective behaviors in a fish-inspired robot swarm. *Science Robotics* 6(50), 2021. DOI 10.1126/scirobotics.abd8668.
   Coordination only through LED emission and vision, with no messages. No leader election.
4. **Mezey, Bastien, Zheng, McKee, Stoll, Hamann, Romanczuk.** Purely vision-based collective movement of robots. *npj Robotics*, 2025. DOI 10.1038/s44182-025-00027-2 · arXiv:2406.17106.
   Leaderless collective motion from raw visual projections alone.
5. **Çelikkanat, Şahin.** Steering self-organized robot flocks through externally guided individuals. *Neural Computing and Applications* 19(6), 2010. DOI 10.1007/s00521-010-0355-y.
   Unidentified "informed" robots steer the flock without communication.

## 2. UAV flocking and leader–follower without GPS or communication

6. **Walter, Staub, Franchi, Saska.** UVDAR system for visual relative localization with application to leader–follower formations of multirotor UAVs. *IEEE RA-L* 4(3), 2019. DOI 10.1109/LRA.2019.2901683.
7. **Petráček, Walter, Báča, Saska.** Bio-inspired compact swarms of unmanned aerial vehicles without communication and external localization. *Bioinspiration & Biomimetics* 16(2), 2020. DOI 10.1088/1748-3190/abc6b3 · arXiv:2303.02989.
8. **Horyna, Krátký, Pritzl, Báča, Ferrante, Saska.** Fast swarming of UAVs in GNSS-denied feature-poor environments without explicit communication. *IEEE RA-L* 9(6), 2024. DOI 10.1109/LRA.2024.3390596.
9. **Schilling, Schiano, Floreano.** Vision-based drone flocking in outdoor environments. *IEEE RA-L* 6(2), 2021. DOI 10.1109/LRA.2021.3062298 · arXiv:2012.01245. Code: github.com/lis-epfl/vswarm.
10. **Schilling, Lecoeur, Schiano, Floreano.** Learning vision-based flight in drone swarms by imitation. *IEEE RA-L* 4(4), 2019. DOI 10.1109/LRA.2019.2935377.
11. **Silveria, de Araujo, Nascimento, Givigi.** Decentralized UAV swarms for ground target protection in GPS- and communication-denied environments. IROS 2026, arXiv:2607.20710.
12. **Pliska, Vrba, Víta, Jiroušek, Walter, Saska.** Towards agile vision-based multi-UAV flight: revisiting state estimation. IROS 2026, arXiv:2609.39611.
13. **Angadi et al.** Vision-based leader-follower formation control for cooperative UAVs in GPS-degraded environments. AI-SIIS 2026, arXiv:2609.01420.

## 3. Distributed leader election under minimal signalling

14. **Angluin, Aspnes, Diamadi, Fischer, Peralta.** Computation in networks of passively mobile finite-state sensors. *Distributed Computing* 18(4), 2006. DOI 10.1007/s00446-005-0138-3.
    Population protocols, including the pairwise-encounter rule L+L→L+F. This is the theoretical model closest to CB-ELE's duels.
15. **Berenbrink, Giakkoupis, Kling.** Optimal time and space leader election in population protocols. STOC 2020. Author PDF: people.irisa.fr/George.Giakkoupis/papers/stoc20le-full.pdf.
16. **Burman, Chen, Chen, Doty, Nowak, Severson, Xu.** Time-optimal self-stabilizing leader election in population protocols. PODC 2021. arXiv:1907.06068.
17. **Gilbert, Newport.** The computational power of beeps. DISC 2015. DOI 10.1007/978-3-662-48653-5_3.
18. **Dufoulon, Burman, Beauquier.** Beeping a deterministic time-optimal leader election. DISC 2018. DOI 10.4230/LIPIcs.DISC.2018.20.
19. **Vacus, Ziccardi.** Minimalist leader election under weak communication. arXiv:2502.12697, 2025.
20. **Dieudonné, Petit, Villain.** Leader election problem versus pattern formation problem. DISC 2010. arXiv:0902.2851.
21. **Flocchini, Prencipe, Santoro (eds.).** *Distributed Computing by Mobile Entities.* Springer LNCS 11340, 2019. DOI 10.1007/978-3-030-11072-7.
22. **Daymude, Gmyr, Richa, Scheideler, Strothmann.** Improved leader election for self-organizing programmable matter. ALGOSENSORS 2017. DOI 10.1007/978-3-319-72751-6_10.
23. **Ongaro, Ousterhout.** In search of an understandable consensus algorithm (Raft). USENIX ATC 2014.
24. **Zuo, Yao, Chang, Zhu, Gui, Qin.** Voting-based scheme for leader election in lead-follow UAV swarm with constrained communication. *Electronics* 11(14):2143, 2022. DOI 10.3390/electronics11142143.

## 4. Emergent leadership and drone flocking references

25. **Couzin, Krause, Franks, Levin.** Effective leadership and decision-making in animal groups on the move. *Nature* 433, 2005. DOI 10.1038/nature03236.
26. **Vásárhelyi, Virágh, Somorjai, Nepusz, Eiben, Vicsek.** Optimized flocking of autonomous drones in confined environments. *Science Robotics* 3(20), 2018. DOI 10.1126/scirobotics.aat3536. Uses GPS and radio.

## 5. Surveys

27. **Chung, Paranjape, Dames, Shen, Kumar.** A survey on aerial swarm robotics. *IEEE T-RO* 34(4), 2018. DOI 10.1109/TRO.2018.2857475.
28. **Coppola, McGuire, De Wagter, de Croon.** A survey on swarming with micro air vehicles. *Frontiers in Robotics and AI* 7:18, 2020. DOI 10.3389/frobt.2020.00018.
29. **Georgiadis et al.** A review of localization and sensing technologies for UAV swarms in SAR missions. *EURASIP J. Adv. Signal Process.*, 2025. DOI 10.1186/s13634-025-01269-w.

## Closest competitors for an experimental comparison

| # | method | why |
|---|---|---|
| [1] | PACNav | implicit leader identification, no communication or GNSS; open source |
| [2] | Choi et al. 2026 | newest learned implicit leader–follower, no communication |
| [7], [8] | Petráček 2020 / Horyna 2024 | leaderless baseline with the same sensing assumptions |
| [25] | Couzin informed individuals | leaderless baseline with k informed agents; cheap to add to this simulator |
| [24] | Zuo et al. 2022 | radio-based election and re-election: shows what CB-ELE gives up or gains without communication |
| [14], [16] | population protocols | theoretical baseline for collision-driven elimination time |

**Positioning gap:** none of the robotic works [1]–[13] elects a single leader without communication, and none of them handles leader failure or re-election. The distributed-computing works [14]–[23] elect leaders under minimal signalling, but they assume abstract schedulers or static graphs, not flying agents.
