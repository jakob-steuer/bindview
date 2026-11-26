# bindview: rules and numbers

What bindview reads, the rules by which it draws, every threshold with its unit, what was measured and on
what, and what it does not do. The [README](../README.md) shows the figures, the [manual](manual.md) says how
to make them, [api.md](api.md) lists every name, and [graph-schema.md](graph-schema.md) is the format of a
layout.

[Inputs](#inputs) · [The plain distance table](#the-plain-distance-table) · [The ligand](#the-ligand) ·
[Figures](#figures) · [Predicted models](#predicted-models) · [Several ligands, several models](#several-ligands-several-models) ·
[The pocket](#the-pocket) · [Colour and the page](#colour-and-the-page) · [Limits](#limits) ·
[Determinism](#determinism) · [Tests and data](#tests-and-data)

## Inputs

### Structure files

| | rule |
|---|---|
| formats | mmCIF (`.cif`, `.mmcif`) and PDB (`.pdb`, `.ent`), each also gzipped (`.gz`). Another extension is read as mmCIF where the text has a `data_` line and `_atom_site.` items, else as PDB. The text is UTF-8. The file is opened for reading only |
| mmCIF | read by the package's own tokenizer: the first data block, and of it `_atom_site`, `_chem_comp` (its `pdbx_smiles`), `_ma_qa_metric` and `_struct_conn`. An `_atom_site` loop that ends inside a row is refused |
| models | `read_structure(path)` reads the first model; `read_structure(path, model=n)` the model the file numbers `n` (`pdbx_PDB_model_num`, the serial of `MODEL`), and a number the file does not have is refused with the numbers it has. `read_models(path)` gives every model as a structure of its own, numbered as the file numbers them where those are numbers and no two are alike, else 1, 2, … in file order |
| alternate locations | of several atoms with one name in one residue, the first in the file is read. Where one position (chain, number, insertion code) holds two residue types as alternate locations of each other, the position is read as the type of its first record, in that type's first alternate location, and the atoms of the other type are not read. A second residue name at the same number without an alternate-location id is kept: that is a number given twice |
| numbers and names | a residue is its chain, its number and its insertion code (`pdbx_PDB_ins_code`; PDB column 27). The chain is `auth_asym_id`, the number `auth_seq_id`, the atom and component names `auth_atom_id` and `auth_comp_id`: the authors' names, which every label, table and figure uses. Where the file has no author's field the `label_` one is read. A residue without either number (a ligand or a water in a file without author numbers) is numbered after the last numbered residue of its chain, in file order |
| elements | `type_symbol`; in a PDB file columns 77–78, else from the atom name (a two-letter element starts in column 13, so `" CA "` is a carbon and `"CA  "` calcium) |
| waters | residues named `HOH`, `WAT`, `DOD`, `H2O`, `D2O`, `TIP`, `TIP3` or `SOL`. A water covers no ligand, is no partner of the plain distance table and holds no ion. Its contacts reach a figure as `water_bridge` rows of a table |
| hydrogens | read and never drawn. They choose between the ways a SMILES fits the atoms, and give bond orders where there is no SMILES |
| B-factor column | kept as the number it is, and read as pLDDT only by the rules of [pLDDT](#plddt-and-the-models-own-scores) |
| links | covalent rows of `_struct_conn` and `LINK` records between atoms of one copy of the molecules (the same symmetry operator); in a PDB file the links of a metal are left out |

**The ligand** is named by its residue name (`"STI"`), `"chain:number"` (`"A:142"`), `"chain:name:number"`
(`"A:HEM:142"`), a pair `("A", 142)` or `("A", "60A")`, or a `ResidueKey`. A name that several residues carry is
refused with the list.

**A chain as the ligand** is named `"P"` or `"chain:P"`. It is looked up by the authors' chain name and, only
where no chain has that name, as `label_asym_id`, which a note then says. Its residues are those with `N`, `CA`
and `C` (or nucleotides whose `O3'` lies within 2.0 Å of the next one's `P`), and whatever is bonded to them
(1.9 Å; 2.3 Å with S, P or Se); never a water. A chain of more than 54 residues is not drawn as a ligand: it is
one side of an interface. Its atoms are named `63/OE1` (atom `OE1` of residue 63), which is how a table names
them. Of two of its heavy atoms less than 1.05 Å apart (a mislabelled hydrogen, a duplicate), the one bonded to
fewer others is left out, and a note says so.

### The table of contacts

A list of `Interaction` records, of rows (any mapping), or the path of a table (`load_interactions`). The
`interactions.tsv` of every single case in `testdata/` is such a table, written by PLIP.

| column | also read as | meaning |
|---|---|---|
| `type` | `interaction_type` | the kind, below. Required |
| `residue_number` | `resnr`, `resnum`, `residue_id` | an integer. Required |
| `residue_name` | `restype`, `resname` | Required |
| `chain` | `reschain`, `chain_id` | may be empty where one chain has that number |
| `insertion_code` | `icode` | |
| `protein_atom`, `ligand_atom` | `protein_atom_name`, `ligand_atom_name` | atom names as in the file; empty for a contact made by a group |
| `protein_atoms`, `ligand_atoms` | | the atoms of a group (a ring, a charged group), separated by commas or semicolons |
| `distance` | `dist` | Å, not negative: atom to atom, or centre to centre for rings and charged groups |
| `angle` | | degrees, as the detector reports it |
| `ligand_is_donor` | | direction of a hydrogen bond or water bridge: `true`/`false` (also `t`, `1`, `yes`, `y`); empty where unknown |
| `beyond_contact_set` | | the residue lies outside the caller's contact definition: drawn muted, its name with a dagger |
| `water` | | `chain:number` of the bridging water |
| `occupancy` | `frequency`, `persistence` | the share of an ensemble that makes the contact, 0 to 1 ([Occupancy](#occupancy)) |
| `confidence` | `contact_confidence` | the contact's own confidence on the pLDDT scale, 0 to 100 |
| `predicted` | | true for a contact through a predicted water |
| `pae` | `predicted_aligned_error` | Å; `pae_draws`: the value of each draw behind a median |

The nine kinds, with the spellings that are read as each (case, hyphens and blanks aside):

| kind | also | | kind | also |
|---|---|---|---|---|
| `hbond` | `h_bond`, `hydrogen_bond`, `hydrogenbond` | | `halogen` | `halogen_bond`, `halogenbond`, `xbond` |
| `hydrophobic` | `hydrophobic_contact`, `hydroph` | | `metal` | `metal_complex`, `metal_coordination` |
| `saltbridge` | `salt_bridge`, `ionic` | | `water_bridge` | `waterbridge`, `water` |
| `pi_stacking` | `pistacking`, `pi_stack`, `pistack`, `pi_pi` | | `covalent` | `covalent_bond`, `covale` |
| `pi_cation` | `pication`, `cation_pi` | | | |

* **Delimiters.** A tab, a comma or a semicolon: the one the header row has most of, and of two that are as
  frequent the earlier in that order (`parse_interactions(text, delimiter=)` names one). Numbers have a decimal
  point. A byte-order mark is passed over; text that is not UTF-8 is refused. Blank lines are skipped, and an
  error names the line.
* **Values not given.** An empty cell, `?`, `.`, `None`, `none`, `nan`, `NaN` and `NA`. `NA` is kept as a name
  where the structure has an atom, a residue or a chain of that name (a haem's nitrogen, sodium).
* **An empty table** (a header and no row) means *analysed, nothing found*: the figure is `empty`.
* **Refused:** an occupancy outside 0 to 1, a confidence outside 0 to 100, a negative distance or PAE, and a set of
  confidences none of which exceeds 1 (that is the 0–1 scale, which is not converted).

**How a row is attached.** A ligand atom that is named is looked up by name. A row of a group without its atoms
(PLIP's π-stacking, π–cation, salt-bridge and metal rows can come so) is attached by geometry: every chemically
possible pair of ligand group and protein group is enumerated, and the pair whose centre-to-centre distance is
within 0.02 Å of the reported one is taken; of two inside that, the chemically canonical one, and several rows
of one residue are assigned together so that no pair is used twice. Each attached row says how
(`dia.interactions[i].method`: `named`, `supplied`, `distance-match`, `approximate`, `nearest`, `unresolved`) and,
by geometry, by how much (`delta`, Å). A row that cannot be attached is not drawn, and the figure is `degraded`.
Measured on the seven single cases: their PLIP tables name the atoms of every group, and with those lists taken
out, all 9 group rows are found again by geometry, within 0.005 Å; 7 on PLIP's atoms, and 2 salt bridges on the
same charged group with its carbon counted in.

A hydrogen bond is drawn from donor to acceptor as the row says, except where the donor it names carries no
hydrogen in the depicted protonation state: then the line has no arrowhead and a warning says why.

### The table of an interface

Tab-separated, as `dump_interface_contacts` writes it and `load_interface_contacts` (and `bindview interface
--contacts`) reads it: `interaction_type`, `chain_a`, `residue_number_a`, `insertion_code_a`, `residue_name_a`,
`atoms_a`, `chain_b`, `residue_number_b`, `insertion_code_b`, `residue_name_b`, `atoms_b`, `distance`, `angle`,
`a_is_donor`, `confidence`, `pae`, `water`, and `pae_draws` where a contact has them. The seven columns without
which a row means nothing are required. `interaction_type` is one of the nine kinds as spelled above, atoms
are separated by commas, `a_is_donor` is `true` or `false`, lines that begin with `#` are passed over, and a
contact can be written from either side.

### A layout

The graph a figure writes (`result.graph.to_json(path)`, `--graph`) is read back by `Graph.from_json` and drawn by
`render(graph=...)` (`--layout`) without a layout being run. Its format, versions 1.0 to 1.4, is
[graph-schema.md](graph-schema.md).

### PAE files

| file | how its rows are known |
|---|---|
| AlphaFold 3's `confidences.json` (or `.json.gz`) | the file names them (`pae`, `token_res_ids`, `token_chain_ids`); they are checked against the structure |
| Boltz's `pae_<name>.npz` | the matrix alone: its rows are the tokens of the model, in file order, one for a standard residue of a polymer (the twenty amino acids, `UNK`, `A`, `C`, `G`, `U`, `N`, `DA`, `DC`, `DG`, `DT`, `DN`) and one for each heavy atom of anything else. A matrix of another size is refused. The model is the structure given, or `<name>.cif` / `<name>.pdb` beside the matrix, by name only |
| the AlphaFold database's `AF-…-predicted_aligned_error_v*.json` (or `.json.gz`) | a list with one object that holds `predicted_aligned_error` (the object alone, and the older form with `residue1`, `residue2` and `distance`, are read too): n × n for the n residues of the model's one chain. Refused where the chain has another number of residues or the model has several chains. A chain cut into parts (`Structure.split`) is still one chain |

Row i, column j is the error of residue j when the model is aligned on residue i. **A pair's value is the smaller
of the two directions.** Several files are the draws of one model: a pair's value is their median, never their
largest, and each draw's value is kept (`pae_draws`). Where a model has copies of a chain, which a model may name
in another order in every draw, the chains of each draw are matched to those drawn by superposition (CA RMSD;
assignments within 0.5 Å of the best fit count as equal, and of those the one that renames fewest chains is
taken; at most 5,040 assignments are tried); draws of such a model without their model files are refused.
Boltz's `pde_*.npz` (the predicted error of a distance, which has no frame) and `plddt_*.npz` are refused where a
PAE is expected. A matrix that has a value for no contact of the figure is refused.

### pLDDT and the model's own scores

| `plddt=` (`--plddt`) | where per-atom pLDDT comes from |
|---|---|
| `"auto"` | the B-factor column, **only where the file declares it a pLDDT**: a row of `_ma_qa_metric` with mode `local` whose type or name says pLDDT; a `[0,1]` variant is multiplied by 100. A declaration of a global value only is not enough |
| `"bfactor"`, `"bfactor01"` | the caller's word that the column is pLDDT on 0–100, or on 0–1 (AlphaFold 2 and ColabFold PDB files cannot declare it) |
| a `Confidence` | values handed in: `Confidence.from_values({"A:570": 96.1}, {"C1": 88.0})`; `scale="0-1"` where they are on 0–1 |
| `None` (`off`) | no confidence |

**The scale is never guessed.** On 0–100 a model has no value below 1, so a column or a set of values in which
nothing exceeds 1 is refused; so are values above 1 stated to be on 0–1, negative values, and values above 100.
In a crystal structure the column is a temperature factor, where high means worse, and it is never read as
confidence: not by its range, not from resolution or occupancy. A residue's value is the mean over its atom
records (`_ma_qa_metric_local` is not read: it is keyed on another numbering). The pocket is the residues with a
heavy atom within 4.5 Å of the ligand; the pocket pLDDT is the mean over all their atoms. A model gives the
atoms of a residue values of their own where, in some residue, two of them differ by more than 0.01.

`model_confidence=` takes ipTM, pTM and ranking score from a mapping, a `ModelConfidence` or a JSON file.
`"auto"` reads AlphaFold 3's `summary_confidences.json` beside `model.cif` (`<job>_summary_confidences.json`
beside `<job>_model.cif`), by name only, and only for a file whose pLDDT is read. `trusted=False` and `flag=`
are the caller's verdict; the figure gets a banner and a frame. bindview applies no threshold of its own to it.

## The plain distance table

`interactions="distances"` (`--distances-only`) makes the table from the positions of the heavy atoms, with
nothing installed beyond bindview: `bindview.distance_table.ligand_contacts(structure, ligand, smiles)` for a
ligand that is one residue, `contact_table(structure, chain, partners)` for a chain that is the ligand, and
`interface_contacts(structure, chains_a, chains_b)` for an interface. It is a table of distances, not an
analysis of interactions.

```python
from bindview import Diagram, load_interactions, read_structure
from bindview.distance_table import ligand_contacts

model = read_structure("testdata/kit_imatinib/model.cif.gz")
smiles = "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C"
rows = ligand_contacts(model, "STI", smiles)                       # the plain table, as rows
plain = Diagram.from_structure(model, "STI", smiles=smiles, interactions="distances")
plip = Diagram.from_structure(model, "STI", smiles=smiles,
                              interactions=load_interactions("testdata/kit_imatinib/interactions.tsv"))
print(sorted({r.method for r in plip.interactions}))              # how PLIP's rows were attached
```

| kind | `ligand_contacts` | `contact_table`, `interface_contacts` |
|---|---|---|
| hydrogen bond | N or O of the ligand within 3.5 Å of N or O of a residue, one able to donate and the other to accept | the same; donors and acceptors by residue and atom name |
| salt bridge | an atom of a charged group within 4.0 Å of an atom of a group of the other sign; one row per pair of groups | the same |
| hydrophobic | two carbons, neither bonded to N or O, within 4.0 Å; the closest pair of a residue | the same, one row per pair of residues |
| π-stacking | two aromatic rings, centres within 5.5 Å, planes within 30° of parallel or of perpendicular, one centre within 2.0 Å of the other ring's axis | only with `stacking=True`: one row per pair of residues; two rings that stack have no hydrophobic row between them |
| π–cation | the centre of a positive group within 6.0 Å of a ring's centre and within 2.0 Å of its axis | none |
| halogen bond | F, Cl, Br or I on a carbon within 4.0 Å of N, O or S; at least 135° at the halogen and 90° to 150° at the acceptor; the closest acceptor of a residue | none |
| metal | a metal within 3.0 Å of N, O or S | none |

* **Two atoms on one metal have no hydrogen bond.** Where an atom of the ligand and an atom of a residue both lie
  within 3.0 Å of one metal (an ion beside the ligand, a haem's iron, a metal of the ligand), they hold it; they
  are about 3 Å apart because it is between them. A pair that is a salt bridge is no hydrogen bond either.
* **What the ligand is** comes from its SMILES: its aromatic rings, its charged groups, which atoms carry a
  hydrogen. Without bond orders (no SMILES, no hydrogens in the file) every N and O may donate and accept, no ring
  is known to be aromatic, and only carboxylates, guanidiniums, phosphates and sulfonates are charged groups.
* **What it lacks.** No hydrogen is placed, so no angle is checked and a hydrogen bond has no angle. No water
  bridge. Partners of `ligand_contacts` are residues within 8 Å that have a template (the twenty amino acids and
  the eight nucleotides) and metal ions: a cofactor, a modified residue or another ligand beside the ligand is
  no partner. The chain table has no π–cation, halogen or metal rows; a metal only takes hydrogen bonds away.
* The thresholds for rings, cations, halogens and metals are PLIP's published defaults (Salentin et al., *Nucleic
  Acids Res.* 2015); the code is not PLIP's.

**Against PLIP on the seven single cases.** A contact is its kind and its residue; PLIP's are the committed
`interactions.tsv`, the plain table is `ligand_contacts` with the case's SMILES (`kit_imatinib`,
`ar_bicalutamide`, `ar_r1881`, `dhfr_methotrexate`, `ca2_celecoxib`, `hb_carbon_monoxide`,
`egfr_osimertinib`; 0.3 s of CPU for all seven):

| kind | PLIP has | the plain table has | both |
|---|---|---|---|
| hydrogen bond | 18 | 19 | 14 |
| salt bridge | 4 | 3 | 3 |
| hydrophobic | 32 | 42 | 32 |
| π-stacking | 2 | 2 | 2 |
| π–cation | 2 | 3 | 2 |
| halogen bond | 1 | 2 | 1 |
| metal | 0 | 1 | 0 |
| water bridge | 5 | 0 | 0 |

* The 4 hydrogen bonds that PLIP has and the table lacks are 3.67 to 4.03 Å long, beyond the table's 3.5 Å. The 5
  that only the table has are 2.54 to 3.33 Å long; PLIP places hydrogens and checks angles, the table does not.
* PLIP's salt bridge in `kit_imatinib` (Asp810 to the piperazine, 5.3 Å from centre to centre) is not in the
  table, which measures from atom to atom: the nearest oxygen is 4.7 Å from the nitrogens. The table's extra
  π–cation (Phe811 and the aminopyrimidine of imatinib, which it counts as a guanidinium-like cation), halogen
  bond (Val746 to a fluorine of bicalutamide) and metal row (the zinc of carbonic anhydrase on celecoxib's
  sulfonamide) are not in PLIP's tables.
* Every hydrophobic contact of PLIP's is in the table, with 10 more: the table prunes nothing.
* Seven cases are a small sample, all crystal structures of drug-like ligands.

## The ligand

**A SMILES laid on the atoms.** The ligand is built from the structure file, so every atom keeps its name. The
atoms are joined by distance (1.3 × the sum of the covalent radii, relaxed in steps to 1.6 × where the SMILES does
not fit at first, which the result says), and the SMILES, stripped to its graph, is matched onto that. A bond
the model has and the SMILES does not is taken out: to a metal it is kept as coordination, otherwise it is a
clash and the result names the two atoms. The built molecule is compared with the SMILES by canonical SMILES;
another molecule is an error (`LigandBuildError`), never a figure. Fragments of the SMILES that the structure
cannot hold (a counter-ion) are dropped where the rest fits exactly. A SMILES that does not fit at all is not
used, and the result says so.

**Where it fits in several ways.** Without bond orders the graph has symmetries the molecule lacks: the ketone of
4-hydroxycyclohexanone fits either oxygen. Of up to 1,000 ways, the one taken is the one whose higher bond orders
lie on the model's shorter bonds (the sum of order times length) and, in a file with hydrogens, that gives each
atom the hydrogens the file has on it (each one off costs 0.1 Å). Another way than the first is taken only where
it is better by more than 0.1 Å for each bond order it moves; where the model does not tell two ways apart, the
result says so. The two oxygens of a carboxylate or a phosphate, the nitrogens of a guanidinium, are one group
either way and stay as the first way has them, unless the file's hydrogens say which carries one.

**A SMILES that bonds a metal like any atom is refused.** The PDB's dictionary writes the haem's iron with four
bonds; laid on `A:HEM:142` of `hb_carbon_monoxide` that gives a nitrogen a valence of 4, and the error says that
the ligand holds a metal and that without a SMILES it is drawn from its connectivity. Without one the haem is
drawn so: 43 atoms, all bonds single, status `degraded`.

**Bond orders from geometry** are inferred only where the file has the ligand's hydrogens. On heavy atoms alone
the method invents multiple bonds, so only the connectivity is drawn, every bond single and no hydrogen shown,
and the figure is `degraded`.

**Drawn as at physiological pH, for acids only.** Carboxylic, phosphoric and sulfonic acids are drawn ionised
(`ionize_acids=False`, `--no-ionize`, draws them as the SMILES has them). Bases are drawn as the SMILES has them,
because their protonation cannot be read from a structure. Stereo is taken from the model and compared with the
SMILES, and a centre that differs is named. P and S centres with two terminal oxygens carry no stereo.

**A metal of the ligand** keeps its coordination bonds as lines of the ligand's drawing. A part of the ligand that
the model binds to nothing else of it (a hydroxide off its cobalt) is drawn apart, and a note gives its distance
to the nearest atom and metal.

**A covalent ligand** (`covalent=True`, `--covalent`). A bond between the ligand and N, O, S or Se of a residue is
taken where the file declares it (`_struct_conn`, `LINK`; at most 2.6 Å) or where the two atoms are within 0.25 Å
of the sum of their covalent radii. It becomes a row of kind `covalent`, drawn as a bond on a highlight; a table
can bring such rows itself. A contact between atoms within three bonds of each other across it is left out. The
ligand is drawn as bound: where the SMILES has a multiple bond at the bound atom and the model has that bond at
the length of a lower order, it is drawn at the lower one. On `egfr_osimertinib`: C9 to Cys797 SG, 1.80 Å,
declared by the file, and the C9–C8 double bond of the SMILES, 1.52 Å in the model, drawn single.

**An ion as a bridge** (`ions=True`, `--ions`). An ion of the model (a residue of one metal atom) that a donor of
the ligand and a donor of another residue both hold is drawn with the ligand, and each residue that holds it gets
a contact with its length. A donor holds an ion within 2.6 Å (Mg), 2.7 Å (Mn, Zn, Fe, Co, Ni, Cu), 2.8 Å (Cd, and
any other metal), 2.9 Å (Ca, Na) or 3.3 Å (K). A hydrogen bond of the table between two atoms that hold one ion is
left out. Where the model has no ion, none is made up. On `ca2_celecoxib` the result says: "ion bridge: Zn is
between N3 of the ligand and His96 NE2, His94 NE2, His119 ND1". With the zinc itself as the ligand (`"ZN"`, no
SMILES, `interactions="distances"`) the figure has three metal contacts, to His94, His96 and His119.

## Figures

### Styles and their settings

A style is a set of settings. Each can be set by hand on `render`, in either style; left out (`None`) it is the
style's own (`bindview.styles.base.STYLE_SETTINGS`; on the command line `--residues`, `--lines`, `--exposure`,
`--lining`, `--distances`).

| setting | `detailed` | `schematic` | what it can be |
|---|---|---|---|
| `residues` | `"atoms"` | `"bubbles"` | `"atoms"`: the part that makes a directional contact as a skeletal fragment, and an eyelash for a residue with hydrophobic contacts only. `"bubbles"`: a bubble in the colour of the residue's class. `"names"`: the name alone. Or residue by residue: `{"Gln712": "atoms", "*": "bubbles"}` |
| `lines` | `"detailed"` | `"schematic"` | colours and dashes of the lines; `"schematic"` draws a hydrogen bond of the backbone solid and one of a side chain dashed |
| `exposure` | `False` | `True` | grey halos behind the ligand's atoms that are open to solvent |
| `lining` | `"none"` | `"contacts"` | the wall of the pocket, below |
| `distance_labels="auto"` writes | hydrogen and halogen bonds, metals, water bridges | nothing | `"all"` (every line but hydrophobic ones), `"none"` |

What is drawn of a residue as atoms: the alpha carbon; the whole side chain where any atom of it makes a contact;
the backbone nitrogen and carbonyl only where they make one. That needs a template: the twenty amino acids (and
force-field names such as `HID`, `CYX`, `ASH`) and the eight nucleotides `A`, `C`, `G`, `U`, `DA`, `DC`, `DG`,
`DT` (and `RA`, `G5`, `DA3` and the like). Any other partner (a cofactor, a modified residue, water) is drawn as
its name, with its atoms in the tooltip and `data-protein-atoms`. `atom_labels="elements"` writes the element on
an atom of a residue; `"names"` its name in the file.

**The lining** (`lining="contacts"`): a stretch of the ligand's outline for each contact, centred where the
straight line from the ligand's atom, ring or group to the residue leaves the outline, 0.7 bond lengths to either
side. Its colour is that of the part of the residue that makes the contact: a carbon or the face of a ring is
hydrophobic; N, O or S polar; a charged group positive or negative; a metal a metal. Where the row does not say
which part, the residue's class. Where two stretches meet, each point is the nearer one's; two of different
colours on one place are moved apart. A predicted water and a contact made only by other models have no stretch.
The key names the colours drawn. `"nearest"` is a band in the class of the nearest residue, open where none is
near; boxes on a plane have no lining.

A line of a salt bridge or a π–cation contact points at the charged group and ends beside it; `charge_halos=True`
shades the group. A line that runs over a word not its own (a letter of the ligand, a residue's name, another
line's distance) is broken around it (`break_lines=False` draws it whole); a line that ends at a word, and a short
line of which less than half would be left, are not broken.

### Layout and legibility

1. **Residues are rigid bodies** tied to the ligand atoms, rings and groups they contact. Most constrained first,
   each is seated at the cheapest of up to 4,000 candidate poses within 7 bond lengths. The cost: link lengths
   against their ideal (a short link costs 4 times a long one), lines crossing ligand bonds or other lines, lines
   over ligand ink, distance from the ligand. Collision is a hard rule, not a cost: at least 5 units between two
   bodies, 3 between a line and a body it does not end on.
2. **Re-placement passes** lift each body and seat it again against the others, up to five times.
3. **Ejection chains:** a body that is still badly placed tries the poses it wants, and whoever is in the way is
   moved; the exchange is kept where the total cost falls.
4. **Polish** nudges every body by shrinking steps.
5. **An independent audit** counts overlaps, residues not placed and lines through a residue they do not end on.
   A layout is *valid* where all three are zero (and boxes on a plane are a faithful picture, below). A layout from
   outside (a graph moved by hand) gets the same audit.

**Legible** is a number: the smallest text must print at `min_font_pt` (5 pt) with the figure fitted to
`figure_width_mm` × `figure_height_mm` (180 × 240 mm). The bond length is tried at 30, 26, 22 and 19 units, the
text sizes fixed, largest first; a bond length that cannot fit by an estimate is skipped. A figure is fitted by
shrinking, never by enlarging, so its smallest text is at most what it is at full size (7.1 pt with the default
`Metrics`): a `min_font_pt` above that is met by no page, and the result says so and names larger fonts
(`metrics=`). A ligand that alone, at 19 units, is too wide for the page is found before any layout is run.

| status | means | what to do |
|---|---|---|
| `ok` | an atom-level figure, valid and legible, nothing left out or assumed | |
| `degraded` | drawn, but a row could not be attached (not drawn) or the bond orders are unknown (no SMILES, no hydrogens) | read `result.warnings`; give a SMILES, or rows the ligand has |
| `empty` | analysed, no contact: the figure shows the ligand, says so in words and lists the residues within 4.5 Å | nothing is wrong; check the table if contacts were expected |
| `unreadable` | not valid, or not legible at the page size. By default a per-residue summary table stands in its place and says why | a larger page (`figure_width_mm`), `style="schematic"`, `detail=None`, or for an interface a smaller site; `on_unreadable="render"` draws it anyway, `"none"` draws nothing |

A file that does not parse, a ligand that is not there, a SMILES that does not parse or that fits the atoms as
another molecule, and detection asked for without PLIP raise a `BindviewError` (`StructureError`, `LigandNotFoundError`, `LigandBuildError`, `InteractionError`,
`DetectionUnavailableError`, `RenderError`, `GraphError`, `EnsembleError`). A graph laid out for another style
raises `ValueError`. Nothing found is never an error, and an error is never an empty figure. `bindview` exits with
0 where a figure was written, 3 where it is not readable at the size asked for, and 2 on an error.

**How much of a residue is drawn: `detail`.** `None` (the default) draws a residue as its style does. `"box"`
draws it as one labelled thing; `"backbone"`, `"sidechain"` and `"atoms"` draw the part that makes the contact and
add N, C and O, the whole side chain, or every heavy atom. Only a residue that makes a directional contact shown by
a line gets atoms; one with hydrophobic contacts only stays a box, an eyelash or a bubble at every level, and a
hydrophobic line ends on the residue. One level for both sides, or `{"ligand": ..., "protein": ...}`; for an
interface one for each participant. Beside a column, a box is the alpha carbon (of a nucleotide, C4'), the side
chain (the base) stands to its side, N above and C and O below where the chain runs (C5', O5' and P above, O3'
below); residues of the column that share a partner stand on one side; no ink of one residue's atoms comes within
6 units of another's, and the column is stretched by the least that takes. No line runs across atoms it does not
end on, nor, for a residue at a level, through a name or label of its own residue. Atoms stand beside the boxes of
a column only: with boxes on a plane a level is refused in words. The schematic style draws bubbles at every level
and says so. No level has a threshold of its own: the figure is narrowed or refused by the rules above.

### A ligand drawn as a chain

`peptide.decide`: a ligand is drawn as a chain of its residues where it has **a run of at least 8 residues, at
least half of them with a proteinogenic side chain**; every other ligand is drawn as atoms, whatever its size.
`ligand_as="atoms"` or `"chain"` overrules it. Both numbers are a choice, not a calibration: 8 because the drawing
of atoms holds about 55 heavy atoms, about seven residues, and a peptidomimetic of four or five units drawn as
boxes would lose the atoms that matter; half because a column of unnamed residues says nothing. A chain that the
file gives as a chain of its own is asked the first condition only (its residues have names), and a chain of
nucleotides is a chain from 3 nucleotides.

* **Residues read or perceived.** A chain of its own in the file keeps its residues, names and deposited numbers;
  its SMILES is written from the residue names (the twenty, common modified residues such as `PTR`, `TYS`, `SEP`,
  `TPO`, caps `ACE`, `NH2`, `NME`), and a component that is not known leaves the bond orders unknown, which the
  result says. A peptide that the file holds as one hetero residue is perceived from its bonds: backbone units
  N–Cα–C(=O), the longest run walked from its N-terminus, each side chain matched exactly against the twenty and
  a short list of others; what matches nothing is `Xaa`, what is in no residue hangs on the residue that carries
  it, drawn as atoms. Positions of a perceived chain count from its N-terminus and are not sequence-database
  numbers; the figure says which numbering it has.
* **A column.** Boxes top to bottom from the N-terminus; the chain does not bend, and its path in space is not
  shown. Every contact, a hydrophobic one too, is a line that ends on the residue whose atom makes it, and no line
  crosses another box. A residue of the protein whose contacts lie more than 8 residues of the chain apart is drawn
  once beside each stretch, and a note names it.
* **More room, not smaller.** A crowded chain is laid out again with its residues further apart, from 0.9 to 3.0
  bond lengths in ten steps, as long as the page still holds it; six layouts at most, then the best is taken for
  what it is. `effort="quick"` gives up early: where the first layout leaves at least max(4, n/10) of n residues
  unplaced after the first round of exchanges, the roomiest layout the page allows is tried, and where that leaves
  at least max(3, n/15) the figure is refused without further tries. A figure that is drawn is the same file
  either way. The two numbers were set on figures of chains (peptides, and stretches of proteins drawn as chains)
  whose data are not part of the package.
* On the open systems: hirudin's ten residues on thrombin (`thrombin_hirudin`, chain H) are drawn `ok` in both
  styles; semaglutide (`glp1r_semaglutide`, chain P, 30 residues) is `degraded`, because the group on its lysine
  (`WF1`) is not a known component.

### RNA and DNA

A nucleotide beside a ligand is a residue: sugar and phosphate its backbone, the base its side chain. The detailed
style draws the part that makes a contact whole: the base with C1' for a contact of the base, the phosphate (five
atoms) for one to the phosphate, the ring of the sugar for a hydroxyl. The schematic style gives nucleotides a class
of their own. PLIP looks at protein unless told otherwise:
`bindview.detect.detect_interactions(structure, ligand, receptor="nucleic")`. A chain of nucleotides is recognised by its atoms (a sugar whose `O3'` is bound to the next `P`), so a
modified nucleotide is one of it; it can be the ligand, drawn 5′ to 3′, and one side of an interface. In the plain
table a nucleotide's donors and acceptors come from its template and its phosphate is a charged group. Name the
protein chains as `partners` of one strand of a duplex: left out, the other strand brings every base pair as rows.

### Base pairs

`BasePairs(structure, [chain_1, chain_2], pairs, contacts=..., ligand=...)` (`bindview basepairs`) draws two strands
as their secondary structures facing each other. The pairs are read, not detected:

| reader | reads |
|---|---|
| `read_mmcif_pairs` | `_ndb_struct_na_base_pair` of a deposited mmCIF; a pair with a crystal mate is not drawn, and for the asymmetric families the file does not say which base uses which edge |
| `read_fr3d_pairs`, `read_fr3d_stacks` | the FR3D tables RNA 3D Hub serves for an entry (CC BY 4.0, which the figure credits); near pairs are left out |
| `read_rnapolis_pairs`, `read_rnapolis_stacks` | the JSON of RNApolis's `annotator` (MIT; run in an environment of its own, never imported) |
| `pairs_from_dot_bracket` | one strand's dot-bracket (`()`, `[]`, `{}`, `<>`); a family only if named |
| `pairs_from_contacts` | a stand-in from the plain table: two nucleotides with two or more base-to-base hydrogen bonds, not stacked, are a pair without a family, drawn dashed |
| `pairs_within_from_table` | the same within one strand: not neighbours, two or more such bonds, each nucleotide in one pair at most, most bonds first |

Where each strand's pairs and the pairs across nest with the strands joined (a duplex), they are laid out as one
secondary structure with a nick; otherwise (a kissing complex) each strand alone, the second turned onto the first by
the pairs across. A pair that does not shape the layout is drawn pale. Symbols are Leontis–Westhof: a plain line for
a cis Watson–Crick G–C or A–U, a dot for G·U, otherwise the edge of each base (circle Watson–Crick, square
Hoogsteen, triangle sugar), filled cis, open trans. A ligand (`ligand="BGD"`) is a box with its name among the
nucleotides it touches, its contacts from `ligand_contacts=` or the plain table. The status is `empty` where nothing
lies between the two chains. The layout has no randomness.

### Interfaces

`Interface(structure, participants, contacts)` (`bindview interface`): two participants, or three in a row whose
middle one touches both; each is one or more chains (`"A"`, `"D+E"`, `"receptor=D+E"`).

* **The spine** is the participant drawn as boxes: of three the middle one; of two the one that is a single chain
  where the other is several, else the one with fewer residues at the interface, else the first. It must be one
  chain. The residues of the others stand around it, with atoms or as bubbles, and the figure says which is which.
* **The site.** At most `max_residues` (30) residues of the spine. With `site=` the residues within `radius` (8 Å)
  of it, the nearest 30; else the whole interface where it fits; else whole contact clusters with the most contacts,
  as many as fit (two residues are in one cluster if they are neighbours in the chain or touch one partner); else the
  30 consecutive residues with the most contacts of the cluster that has the most. The figure states the site and
  its share. Where a figure is not legible and no site was named, it narrows to three fifths of the residues (never
  below eight), twice at most.
* **The column's order** (`order="auto"`): by sequence, or along the long axis of the interface where that spreads
  the contacts of the residues around it over fewer boxes.
* **Stretches.** Residues of the spine are one stretch where no more than 3 residues not at the interface lie
  between one and the next. Residues that the file leaves out (a loop not modelled) count as lying between their
  neighbours; numbers that jump between two residues bonded to each other count as none.
* **The plane** (`layout="plane"`, or `"auto"` for the rule): each box where its residue lies on the least-squares
  plane through the residues' centres, seen from the partner. Two residues are neighbours within 8 Å and apart
  beyond 10 Å. A projection is drawn only where it keeps at least 90 % of the neighbour relations and its
  distances are off by at most 10 % (stress); both numbers are written in the figure with a scale bar, and a
  projection that fails is not drawn, even where asked for. The rule of `"auto"`: a column for fewer than three
  residues or one stretch, otherwise the plane where it passes. The default is the column, and `layout_rule` says
  what the rule would have chosen. Only the spine is a map: the other side stands where there is room.

**What the plane's thresholds mean in terms of shape.** `tests/data/plane-calibration.json` is what
`scripts/calibrate_plane.py` writes (run again on this tree, the same file): points one per 36 Å², at least 4.3 Å
apart, roughened by 1.5 Å, on surfaces that turn by a given angle from their middle to their rim, 300 sets a row.
The share of sets that pass both thresholds:

| surface, 20 and 30 residues | 0° to 45° | 60° | 75° | 90° | 105° | 120° | 150° |
|---|---|---|---|---|---|---|---|
| spherical cap | 100 % | 98–99 % | 63–69 % | 6–12 % | 0 % | 0 % | 0 % |
| cylinder | 100 % | 100 % | 96–98 % | 77–79 % | 34–51 % | 6–22 % | 0–4 % |

A surface that turns by up to about 60° is drawn; a hemisphere is not. With 6 to 12 residues the share of
neighbours kept hardly moves (there are few pairs) and the distance error decides. The thresholds are explicit and
calibrated on shapes, not on readers.

## Predicted models

### Confidence

| band | pLDDT | by weight (default) | `plddt_colors=True` |
|---|---|---|---|
| very high | 90 and above | full stroke | `#0053d6` |
| confident | 70 to 90 | full stroke | `#65cbf3` |
| low | 50 to 70 | 0.7 × the width, opacity 0.62 | `#ffdb13` |
| very low | below 50 | 0.5 × the width, opacity 0.42, and `?` | `#ff7d45` |

A contact is drawn at the **lower** of its two ends: the ligand atom's pLDDT (a ring's or a group's: the mean of its
atoms) and the residue's. Weight, opacity and mark are three channels so that the figure survives greyscale; in the
schematic style a bubble's outline is solid, dashed or dotted. An atom drawn of a residue is at its own value where
the model gives atoms values of their own (`plddt_by_atom=False`: at its residue's), and a footnote says which.
`plddt_colors=True` fills residues and sets every ligand atom on the colour of its band, draws no line weaker, and
shows all four bands in the key; `"bonds"` colours the bonds half by half. The colours are AlphaFold's and the same
in every scheme and theme. ipTM, pocket and ligand pLDDT stand under the title and in `result.confidence`. Which
contacts exist, their kinds and lengths are the same with confidence on and off; `confidence=False` draws the
plain figure. bindview computes no confidence of its own and applies no threshold that judges a model.

### PAE on contacts

A contact carries the PAE between its two residues (in a ligand diagram, between its residue and the ligand atoms
that make it; of a chain ligand, its residue). In an interface it is written on the line as `PAE 11`, rounded to a
whole ångström: no tag below 5, a bold tag from 10, the range under the title, and a sentence where every contact is
at 10 or more. In a ligand diagram (`Diagram.from_structure(pae=...)`, `--pae`), and with `pae_colors=True` in an
interface, it is a glow behind the line: `#0053d6` below 5 Å, `#65cbf3` to 10, `#ffdb13` to 15, `#ff7d45` beyond.
PAE never changes a stroke, which belongs to pLDDT, and the two are never combined. The thresholds are the
customary reading of PAE, not a calibration.

**ipSAE** (`Interface(..., ipsae=True)`, `--ipsae`): a score of the interface as a whole, from the matrix, in both
directions. For an aligned residue i of one side, the residues j of the other with PAE[i][j] below the cut-off
(10 Å, or the number given) are averaged as 1 / (1 + (PAE[i][j] / d0)²), with d0 = max(1, 1.24 · (n − 15)^(1/3) −
1.8) Å for those n residues (1 Å for fewer than 27; 6.26 Å for 290). The score of a direction is the largest such
average, 0 where no pair is below the cut-off. It is the definition of its authors (Dunbrack 2025), in the variant
their script reports as ipSAE. A residue enters by its alpha carbon (a modified residue by the row of its CA), so a
ligand, an ion or a nucleotide does not; several chains of a participant are one side; draws give the score of each
and their median. No threshold and no verdict is attached.

| open interface (plain table) | contacts | PAE of the contacts | pairs tagged, bold | ipSAE |
|---|---|---|---|---|
| `afdb_fah`, two domains of one chain | 13 | 1 Å | 0, 0 of 12 | 0.89 / 0.92 |
| `afdb_tectb`, the same | 7 | 3 to 13 Å | 5, 3 of 6 | 0.05 / 0.11 |
| `afdb_apobec3d`, the same | 5 | 13 to 19 Å | 4, 4 of 4 | 0.02 / 0.02 |
| `tnfa_trimer`, chains A and B, median of two draws | 27 | 0.4 to 9.2 Å | 5, 0 of 25 | 0.91 / 0.91 |

The three AlphaFold database models are each one chain cut at its domain boundary: PAE between two parts of one
chain, where residues that follow each other have a low PAE whatever the domains do. A figure with ipSAE says so
where its two sides are parts of one chain.

## Several ligands, several models

### Comparisons

`Comparison(diagrams)` (`bindview compare`): ligands of one pocket against one frame of residues, which never moves
between panels. The difference view (two ligands) classifies each (residue, kind) as both, only this one, or only
the other; the consensus view puts the share of ligands on every residue.

* **The common core** is RDKit's maximum common substructure, seeded with that of the Murcko scaffolds: elements
  match, and bond orders, each bond its own kind (an aromatic bond is no single bond); where a ligand is known by
  its connectivity alone, an aromatic bond of another matches its single one. An atom of a ring matches one of a
  ring, and a ring is in the core whole or not at all. Each search stops at 30 s, and a result cut short is marked
  incomplete.
* **Aligned on the core** only where it covers at least 60 % of the smallest ligand's heavy atoms and at least six
  atoms (`min_fraction`, `min_atoms`). The figure states the numbers and the decision either way.
* **Three frames.** With a core: the core on the same coordinates in every drawing. Without, where the pockets are
  one pocket: each ligand turned to its own pose (difference) or the footprint of the poses (consensus). Without
  either: residues in sequence order and no geometry claimed; the difference view is then `degraded`. Two structures
  are one pocket where, of the residues with a backbone atom within 10 Å of either ligand, at least four and at
  least 70 % have the same name in both, and their backbones superpose within 3 Å RMSD.
* Residues are matched by chain, number and insertion code; one that two structures name differently (a point
  mutant) is one residue, and a warning names it.

On the open sets: the four corticosteroids of `gr_steroids` share a core of 24 of the smallest ligand's 28 atoms
(86 %) and are aligned on it; `gr_diverse` and `ar_ligands` have no core that passes and are drawn in their
superposed pocket (pocket RMSD at most 0.83 Å in the three sets).

### Ensembles

`Ensemble.from_structures(models, ligand, smiles, interactions)`: one complex in several models, one table each.

* **One contact across models** is one residue, one kind and the same ligand atoms, where atoms that the symmetry of
  the molecule exchanges count as one (the fluorines of a CF3, the oxygens of a carboxylate), a hydrophobic contact
  is matched on its residue alone, and the protein atom, the direction of a hydrogen bond and the water are not part
  of it. Its share is a count over the tables.
* **The model drawn** is the medoid: the one whose ligand lies nearest to the others' by heavy-atom RMSD after the
  pockets are superposed, without fitting the ligand and up to its symmetry (`member=` names another). Every line
  is a contact of that model with its own distance; a contact it does not make is named under the figure, or drawn
  faint with `diagram(absent="faint")`, and is still not counted.
* **All the models in one figure** (`consensus(least=0.25, atoms=0.5)`): a residue counts in every model that has
  any contact with it. From half of the models on it is drawn as atoms (where a line of it has a direction), each of
  its sites with one line to the part of the ligand (a ring system, an end group, a piece of four atoms or more)
  that most of its models touch there, where at least half of its models do; any other residue is a ring alone, and
  below a quarter it is left out and named. No single model is drawn, so there are no distances, no buried
  fraction, no confidence.
* Chains are taken as named, and residues of the pockets must match. Every two models are superposed, so the time
  grows with the square of their number.

### Occupancy

`occupancy` on a row is the share of an ensemble that makes the contact, 0 to 1; `ensemble=` names the ensemble for
the key. By default (`occupancy_style="steps"`) a line is drawn at full width and opacity from 90 %, at 0.7 × the
width and opacity 0.8 from 50 %, and at 0.38 × and 0.5 below, with the share written where its distance would stand
(`occupancy_labels="partial"`: only below 100 %). `"continuous"`: width 0.4 + 0.6 × share, opacity 0.35 + 0.65 ×
share, nothing written. A residue is drawn at its most persistent contact. Occupancy and confidence both scale the
stroke and are never multiplied into one number; the key and `data-occupancy`, `data-plddt` keep them apart.

## The pocket

### Buried fraction

`buried_fraction = 1 − SASA_complex / SASA_free`, by Shrake–Rupley on heavy atoms (probe 1.4 Å, 240 points an
atom); everything in the model but water covers. It is in `result.burial` and `--json` always, and under the title
unless `burial=False`. Below **0.70** (`exposure.BURIAL_WARNING_BELOW`) the result and the figure say that most of
the ligand's surface is open: the site may be incomplete in this model (one chain of an oligomer, a missing
cofactor), or the ligand may not sit in a pocket.

* **What it was calibrated on:** 64 AlphaFold 3 co-folds of one protein chain with one ligand, 32 with the site
  inside one subunit and 32 with a site between subunits or one that needs a cofactor (16 targets). Those models are
  not part of the package.
* **What it means:** the model as given leaves the ligand open. It says nothing of why.
* **What it does not mean:** a well-buried ligand is no evidence that the site is complete; a ligand that drifted
  into another groove of an incomplete model is buried too. It is a property of the model, not a score.
* **Where it applies:** to a ligand drawn as atoms that is not a chain. A peptide or a strand lies along a surface:
  its fraction is reported and no warning is drawn (`warn_below` is `None`; the hirudin peptide on thrombin is 41 %
  buried). An interface and a figure of several models have none.
* **On crystal structures:** none of the 85 ligands of the Astex Diverse Set is below the cut (lowest 0.704), and
  the seven single cases lie between 0.72 and 1.00.

### Predicted solvent sites

`render(solvent_sites=True)` (`--solvent-sites`) predicts waters that could bridge ligand and protein, off unless
asked for. Lattice points (0.5 Å) within 3.5 Å of a ligand N or O, no closer than 2.6 Å to an N or O; at least 60 %
of 30 directions run into the model within 8 Å; scored by the N and O of both sides within 3.3 Å that point at the
point (within 100° of the open side of an atom with one bonded heavy atom, 65° with more); best first, at least
2.6 Å apart, six at most; a partner on each side. A site is drawn only towards a residue that the ligand atom does
not already contact. It is never predicted in a structure that has waters of its own. It is no contact: not counted,
not in a fingerprint, and an empty figure stays empty. It is marked as predicted wherever it appears (`pred. H₂O`, a
dotted line, the key, `data-predicted`, a `solvent_site` node with `predicted: true`).

**The hold-out** (`tests/data/solvent-validation.json`, what `scripts/validate_solvent.py` writes; run again on this
tree, the same file): waters were taken out of the 85 structures of the Astex Diverse Set, and the predicted sites
compared with them. A bridging water is one within 3.5 Å of a ligand N or O and of another N or O; a site matches a
water within 1.5 Å; chance is the same number of sites placed at random where a water has room. The structures are
sorted by name, and the 42 at even positions are the held-out half; no parameter of the predictor is set on them:

| set | bridging waters recovered | by chance | sites with no water within 1.5 Å | by chance |
|---|---|---|---|---|
| held out, 42 structures | 52 of 73 (71 %) | 41 % | 54 of 118 (46 %) | 58 % |
| held out, resolution 2.0 Å or better, 23 | 32 of 43 (74 %) | 38 % | 24 of 64 (38 %) | 56 % |
| all 85 | 110 of 156 (70 %) | 42 % | 109 of 245 (44 %) | 57 % |

Every figure with a predicted site prints the first row as a sentence (`bindview.solvent.VALIDATION`).
`tests/data/solvent-validation.targets.json` is a second set of 29 structures on eight targets (18 of 23 bridging
waters recovered, 18 of 42 sites without a water). **What the numbers do not say:** they are on crystal structures,
in which the gap a water left is still there, which is why chance does well. A predicted model is not shaped around
a water and need not leave the gap; there the predictor is untested, and these numbers are an upper limit. Symmetry
mates were not built. A site with no modelled water is not proven empty. Sites held by few partners are the ones to
doubt: in the half that sets no headline, a water lies on 22 of 47 sites with two partners and on 28 of 39 with four
or more. Ion sites are not predicted.

## Colour and the page

**Colour schemes.** `colors="standard"` or `"contrast"`; the theme (`"light"`, `"dark"`, `"auto"`) picks the
palette, and `"auto"` puts both into one file behind `prefers-color-scheme` or a `data-theme` attribute. A
`Palette` of your own takes no scheme. `"contrast"` changes the classes of residues (and with them the lining), the
ground and the text, and what is drawn of a residue; lines keep their colours, and the layout is the same. Its class
fills are further apart, measured as the OKLab distance × 100 of the worst of all pairs, after simulating colour
blindness (Machado, Oliveira and Fernandes 2009, full severity):

| theme | full vision | protanopia | deuteranopia | tritanopia |
|---|---|---|---|---|
| light: standard, contrast | 11.1, 14.6 | 7.7, 7.0 | 5.2, 9.3 | 6.9, 10.8 |
| dark: standard, contrast | 8.6, 12.8 | 6.3, 11.3 | 3.0, 10.3 | 4.7, 5.0 |

For protanopia alone the light contrast scheme is a little closer than the standard one. A residue's name stands on
every fill of the scheme at 4.5:1 or better, and what is drawn of a residue is as far from the nearest line colour
as in the standard scheme or further (light 8.0 and 9.1, dark 10.4 and 10.8). A residue also carries its name, from
which its class follows.

**Figures that answer the pointer** (`interactive=True`, `--interactive`): a style sheet in the SVG and no script.
Pointing at a residue or a contact keeps it, its contacts and what they reach, and steps the rest back; a click holds
that, and the tab key goes from residue to residue. A page can ask for the same with `data-active="712"` on the root
(the number `data-residue-number` gives). The figure of a ligand drawn as atoms also shows the atoms of a residue
drawn without them while it is pointed at. Taken out of a browser (an `<img>`, a PDF), the file is the plain figure.
For a page with several figures: `stylesheet="inline"` (the sheet in the figure declares the colours for the whole
page), `"scoped"` (for figures in the same colours alone) or `"external"` (in `result.css`, and
`bindview.stylesheet(results)` writes each rule once). Every figure carries `data-residue`, `data-type`,
`data-ligand-atoms`, `data-protein-atoms` and `data-distance` on its contacts.

**Arranging by hand.** `dia.arrange()` (`bindview render --arrange`) serves a page on 127.0.0.1 under an address that
cannot be guessed, where residues and the key are dragged, turned and flipped; each time one is let go the figure is
drawn again from the moved graph, so that lines, eyelashes and the lining follow. `dia.arrange("arrange.html")`
writes the page as one file, which has no bindview behind it and shows a preview only. Both hand back a `Graph`, as `graph.move`, `rotate` and `flip` do; a graph drawn again
(`render(graph=...)`, `--layout`) gets the layout audit, and an untouched one gives the same file.

**The report** (`bindview report`) looks at a model or a folder and draws what it holds. Its rules are the docstring
of `bindview.report`: among them, contacts come from a table beside the model, else from PLIP where it is installed,
else from the plain table, and a detailed figure that is not readable is drawn again in the schematic style.

**The words.** `bare=True` leaves off the confidence line, warnings and notes; they are in `result.notes`, and
`--notes-file` writes them. A fingerprint bit is (chain, number, insertion code, residue name, kind), ordered by
chain, number, insertion code and kind.

### Measured on the seven single cases

PLIP's tables, default page and options, CPU time of the first render (a laptop):

| case | ligand atoms | residues | detailed | units | schematic | units |
|---|---|---|---|---|---|---|
| `kit_imatinib` | 37 | 13 | ok, 1.3 s | 664 × 435 | ok, 1.1 s | 598 × 400 |
| `ar_bicalutamide` | 29 | 9 | ok, 0.7 s | 459 × 514 | ok, 0.5 s | 430 × 399 |
| `ar_r1881` | 21 | 8 | ok, 0.6 s | 607 × 364 | ok, 0.4 s | 372 × 366 |
| `dhfr_methotrexate` | 33 | 10 | ok, 1.1 s | 510 × 624 | ok, 0.7 s | 574 × 428 |
| `ca2_celecoxib` | 26 | 5 | ok, 0.3 s | 339 × 406 | ok, 0.3 s | 311 × 483 |
| `hb_carbon_monoxide` | 2 | 1 | ok, 0.1 s | 186 × 202 | ok, < 0.1 s | 220 × 185 |
| `egfr_osimertinib` | 37 | 7 | ok, 0.7 s | 591 × 356 | ok, 0.6 s | 516 × 371 |

Every one at a bond length of 30 units, smallest text 7.1 pt at 180 mm, no overlap, no residue unplaced, no line
through another residue. With the plain distance table in place of PLIP's, all fourteen figures are `ok` too.

## Limits

* **One ligand a diagram:** one residue, or one chain of amino acids or nucleotides. Several residues that are no
  chain (a glycan) are not one ligand. A chain of more than 54 residues is not drawn as a ligand.
* **No detection of its own.** Contacts come from the caller, from PLIP (optional, GPL-2.0, run by
  `bindview.detect` and imported nowhere else), or from the plain distance table, which is cruder than PLIP.
* **Fragments** for the twenty amino acids and eight nucleotides only; anything else is drawn as its name.
* **Bond orders are not guessed:** without a SMILES and without hydrogens only the connectivity is drawn.
* **Bases are not protonated** for display; only acids are ionised.
* **No trajectory is read.** Occupancy arrives as data; frames written out as files can be an `Ensemble`.
* **An interface** is drawn around one chain; a spine of several chains and more than three participants are
  refused. It has no buried surface. Atoms beside boxes stand in a column only. A participant has one level of
  detail; a single residue cannot be given one by hand.
* **Comparisons** draw residues as nodes and ligands as atoms, take the numbering to be the same in all structures,
  do not draw predicted waters, absent residues or broken lines, and are not drawn again from their graph.
* **PAE thresholds** (5 and 10 Å) are customary, not calibrated; pair ipTM is not printed, PDE is not read.
* **The figure shows what the model contains.** A co-fold of one chain has no partner subunits and no cofactors;
  neither pLDDT nor ipTM shows that, and the buried fraction shows it only where the ligand is left open.
* **Predicted waters** are validated on crystal structures only; ion sites are not predicted.
* **Not built:** 3D rendering, pharmacophores, a viewer. The sketches under `prototype/` are not part of the
  package and are not tested as it is.

## Determinism

A figure has no randomness: no random numbers, no time stamps, no generated ids, a fixed order of attributes and
one number format, and text widths from a built-in table of Helvetica's advance widths, not from installed fonts.
It does not depend on `PYTHONHASHSEED`, and nothing is fetched while drawing. The 2D coordinates of a ligand are
RDKit's CoordGen, which differ in the last digits between platforms and between RDKit releases, and the layout
follows them. So **the bytes are promised for one platform and one RDKit release**: those that
`tests/golden/VERSIONS.json` records (RDKit 2026.09.1, NumPy 2). There every stored figure is compared byte for
byte; elsewhere the same figures are drawn and must hold the same things: the status, the residues, the contacts by
kind and the ligand's atoms.

## Tests and data

```bash
python -m pytest                                  # everything that the files at hand allow
python -m pytest --runslow                        # with the tests marked slow
python -m pytest tests/test_golden.py --update-golden   # store the figures again, after looking at them
mypy                                              # the type check, configured in pyproject.toml
python scripts/fetch_structures.py                # what is on disk against the lists and checksums
python scripts/fetch_structures.py --download     # fetch what is missing from the RCSB PDB
```

PLIP is optional: a test that runs it is skipped where it is not installed. A test that reads a fetched structure is
skipped where the file is absent. `fetch_structures.py` reads three lists (`scripts/solvent_validation_set.tsv`
into `testdata/astex/`, `testdata/waters/manifest.json`, and the `fetch` of each
`testdata/ensemble/*/manifest.json`), fetches only with `--download` (`--only astex|waters|ensemble` for one set),
and puts a file in place only where it is whole and has the checksum that `testdata/structures.sha256` records
(`--revised` takes one whose checksum differs, `--record` writes the checksums of what is on disk).
`scripts/build_api.py` writes [api.md](api.md) from the docstrings.

**The test systems.** Every committed system is a folder of `testdata/` with its model and a `meta.json` that says
what it is, where it is from and under which terms; the single cases also have PLIP's `interactions.tsv`.

| slug | what | entry | licence |
|---|---|---|---|
| `kit_imatinib` | c-Kit kinase domain with imatinib | PDB 1T46 | CC0 |
| `ar_r1881` | androgen receptor with the steroid R1881 | PDB 2AO6 | CC0 |
| `ar_bicalutamide` | androgen receptor (W741L) with R-bicalutamide | PDB 1Z95 | CC0 |
| `dhfr_methotrexate` | human DHFR with methotrexate and NADPH | PDB 1U72 | CC0 |
| `ca2_celecoxib` | carbonic anhydrase II with celecoxib on its zinc | PDB 1OQ5 | CC0 |
| `hb_carbon_monoxide` | carbonmonoxy haemoglobin | PDB 2DN3 | CC0 |
| `egfr_osimertinib` | EGFR kinase domain with osimertinib on Cys797 | PDB 6JXT | CC0 |
| `gr_dexamethasone` and the other three members of `overlay/gr_steroids` | glucocorticoid receptor with four corticosteroids | PDB 1M2Z, 4P6W, 5NFP, 4UDD | CC0 |
| `gr_azd9567`, `gr_azd5423` | the same receptor with two non-steroidal ligands | PDB 6EL9, 5NFT | CC0 |
| `overlay/gr_steroids`, `gr_diverse`, `ar_ligands` | sets of the cases above | | |
| `thrombin_hirudin` | thrombin with a hirudin peptide (chain H, a sulfotyrosine) | PDB 2UUF | CC0 |
| `glp1r_semaglutide` | GLP-1 receptor with semaglutide (chain P) | PDB 7KI0 | CC0 |
| `mhc_peptide` | HLA-A2 with a nonamer (chain P) | PDB 3MRE | CC0 |
| `fab_lysozyme` | HyHEL-10 Fab on lysozyme | PDB 3D9A | CC0 |
| `repressor_dna` | lambda repressor on its operator DNA | PDB 1LMB | CC0 |
| `u1a_rna` | U1A on an RNA hairpin | PDB 1URN | CC0 |
| `rna_kissing` | HIV-1 dimerisation-site kissing complex | PDB 1XPF | CC0 |
| `interface_pae/afdb_fah`, `afdb_apobec3d`, `afdb_tectb` | AlphaFold models of one chain each, with their PAE files | AFDB P16930, Q96AK3, Q96PL2 (v6) | CC BY 4.0 |
| `interface_multimer/tnfa_trimer` | TNF-α as a trimer predicted with Boltz: two of five draws, the scores of all five | made for the package | the package's (MIT) |
| `nucleic/riboswitch_bisguanidine` | a guanidine-II riboswitch dimer with a bivalent ligand, one frame of a simulation | the simulations of [Steuer et al. 2024](https://doi.org/10.1093/nargab/lqae132); its `meta.json` | MIT |
| `md/riboswitch_bisguanidine` | contact tables and drawn frames of three simulations of the same system | its `manifest.json` | MIT |
| `ensemble_bioemu2` | 20 samples of adenylate kinase with AP5A from BioEmu 2, trimmed | made for the package | its `README.md` |
| `ensemble/maob_fad` | contact tables of four MAO-B structures with FAD; the structures are fetched with `waters` | PDB 2C65, 4CRT, 7ZW3, 9FJT | CC0 |
| `ensemble/dhfr_mtx` | contact tables of the 21 models of an NMR bundle; the bundle is fetched | PDB 1AO8 | CC0 |
| `astex` (fetched) | the 85 structures of the Astex Diverse Set, for the water hold-out | `scripts/solvent_validation_set.tsv` | CC0 |
| `waters` (fetched) | 29 structures on eight targets, the second water set | `testdata/waters/manifest.json` | CC0 |

Crystal structures carry no confidence. Where a test needs a pLDDT on a pocket, `tests/conftest.py` makes one up
(`invented_plddt`: 100 minus the temperature factor, between 25 and 99, declared as a model declares it) and says so;
real confidence is in the three AlphaFold database models and the trimer.
