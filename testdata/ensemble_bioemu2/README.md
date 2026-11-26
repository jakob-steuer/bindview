# An ensemble by BioEmu 2: adenylate kinase with AP5A

`ake_ap5a.pdb.gz`: 20 samples as the models of one file, trimmed by `trim.py` to the residues that come within 10 Å
of the ligand in any sample. Data to show how bindview draws an ensemble; **predictions, not measurements**, and not
weighted or selected.

* Made with BioEmu 2 (microsoft/bioemu2, commit 8d4a2b7238716a94543ceaf411dfe32f99c1ae7c; checkpoint `bioemu2-preview`;
  code and checkpoint under the MIT licence) from the project's own example input (`examples/simulate_a_trajectory.py`,
  sequence of 1AKE and AP5A): equilibrium sampling, 200 steps, step scale 1.0, seed 0, batches of 4. The alignment came
  from api.colabfold.com.
* All 20 pass BioEmu's own `check_physicality`. The ribose carbons have the configuration of the chemical component
  in all of them; in 8 samples one to three P–O bonds are 1.33 to 1.41 Å long (1.61 is right); in sample 14 Lys157 has
  an inverted alpha carbon (it is not near the ligand).
* BioEmu 2 writes no confidence: the B-factors are 0 and none is read.
* The samples as BioEmu wrote them are not in the repository; `trim.py` is how this file was cut from them.
