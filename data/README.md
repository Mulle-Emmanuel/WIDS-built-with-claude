# Data
1. Request AWID3: https://icsdweb.aegean.gr/awid/ (registration workflow). Read the terms of use.
2. Put the `Deauth` and `Normal` pcaps under `data/raw/` (git-ignored). Use the pcaps, not the CSV edition.
3. Copy the Wireshark filter published for the Deauth attack on the AWID3 page and pass it to
   `wids extract --attack-filter`. Verify that it matches frames before training.
4. Own captures: only on your own equipment/network. Label with `--attack-macs` if the attacker
   transmitter MAC is not spoofed, otherwise use a display filter.
