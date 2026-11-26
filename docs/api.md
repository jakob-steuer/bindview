# API

Written by `python scripts/build_api.py` from the library itself. In Python: `bindview.options()`, `help(bindview.Diagram)`. How things are done: the [manual](manual.md).

[What goes in](#what-goes-in) · [Options of render](#options-of-render) · [A ligand in its pocket](#a-ligand-in-its-pocket) · [Contacts](#contacts) · [Structures](#structures) · [The layout](#the-layout) · [Several models, several ligands](#several-models-several-ligands) · [Chains](#chains) · [Confidence and colours](#confidence-and-colours) · [Errors](#errors) · [Functions of modules](#functions-of-modules)

## What goes in

`Diagram.from_structure(structure, ligand, smiles=None, interactions=None, *, name=None, ionize_acids=True, plddt='auto', model_confidence='auto', trusted=None, flag=None, ensemble=None, ligand_as='auto', detail=None, covalent=False, pae=None, ions=False)`

| | |
|---|---|
| `structure` | Path to an mmCIF or PDB file, or an already parsed `Structure`. |
| `ligand` | Which residue is the ligand: a residue name (`"LIG_B"`), `"chain:number"`, or a `(chain, number)` tuple. A peptide ligand that is a polymer chain of its own, as in a deposited structure, is named by its chain: `"P"` or `"chain:P"` (`bindview.structure.Structure.resolve_ligand`). Its residues, their names and their numbers are then those of the file. |
| `smiles` | The ligand's SMILES. Optional but strongly recommended: bond orders cannot be recovered reliably from coordinates alone. When omitted, `_chem_comp.pdbx_smiles` from the file is used if present. |
| `interactions` | The contacts: a list of `Interaction` (or dict-like rows), or the path of a table (`bindview.load_interactions`). `"distances"` makes the table from the positions of the heavy atoms (`bindview.distance_table`), which needs nothing installed and is cruder than a detector. `None` means *detect them*, which needs the optional PLIP extra. An empty list means "analysed, nothing found" and yields an `empty` result. |
| `plddt` | Where per-atom confidence comes from. `"auto"` (default) reads the B-factor column **only if the file declares it to be a pLDDT** (`_ma_qa_metric`, as AlphaFold-style models do) and otherwise draws no confidence at all: in a crystal structure that column is a temperature factor, and the value range is never used to guess. `"bfactor"` is the caller's assertion that the column is pLDDT on the 0-100 scale (AlphaFold 2 / ColabFold PDB files, which cannot declare it), `"bfactor01"` that it is pLDDT on the 0-1 scale. The scale is never guessed: a column in which nothing exceeds 1 is refused as 0-100, one in which something does as 0-1. A `bindview.confidence.Confidence` supplies values explicitly; `None` switches confidence off. |
| `model_confidence` | Model-level numbers to display (ipTM, pTM, ranking score): a `bindview.confidence.ModelConfidence`, a mapping, or the path of an AlphaFold 3 `summary_confidences.json`. `"auto"` picks up that file when it sits next to the model under its standard name. |
| `trusted`, `flag` | The caller's verdict. `trusted=False` marks the figure as coming from a model that should not be read as a result; `flag` is the verdict in the caller's words (`"grade C"`). bindview never decides this itself. |
| `ensemble` | What the `occupancy` of the interaction records is a share of, in words for the legend: `"1,000 frames of a 100 ns trajectory"`. Occupancy itself comes with the records; nothing here reads a trajectory. |
| `ligand_as` | How the ligand is drawn. `"auto"` (default) decides by what the molecule is: a ligand with a chain of at least eight amino-acid residues is drawn as that chain, every other ligand as atoms, whatever its size (`bindview.peptide.decide`). `"atoms"` and `"chain"` overrule it. |
| `detail` | How much of a residue is drawn where it is drawn with atoms: `"box"` (no atoms: its name), `"backbone"`, `"sidechain"` or `"atoms"`. One level for both sides, or a mapping with one for the `"ligand"` (where it is drawn as a chain) and one for the `"protein"`. Only a residue that makes a directional contact is drawn with atoms: the part that makes the contact, and what the level adds to it. `None` (default) asks for no level: a residue of a ligand drawn as a chain is its box, and a residue of the protein is drawn as the style draws it. Not `ligand_as`, which says how the ligand is read. |
| `covalent` | Look for covalent bonds between the ligand and a residue, as the file declares them and by distance, and draw them (`bindview.covalent`). Off unless asked for; a row of the kind `"covalent"` in `interactions` is drawn either way. |
| `pae` | The model's matrix of predicted aligned errors (`bindview.pae`), as an `bindview.interface.Interface` takes it: AlphaFold 3's `confidences.json`, Boltz's `pae_*.npz`, or several. Each contact then has the PAE between its residue and the atoms of the ligand that make it (of a ligand that is a chain, the residue of it), drawn as a glow behind its line (`render(pae_colors=...)`). |
| `ions` | Draw an ion that the model has between the ligand and the protein as the bridge it is, with the atoms of either that hold it (`bindview.ions`). Off unless asked for; where the model has none, none is predicted. |

## Options of render

`Diagram.render(style='detailed', theme='light', *, metrics=None, graph=None, **options)`

`style` is `"detailed"` or `"schematic"`; `theme` is `"light"`, `"dark"`, `"auto"` or a `Palette`. `render_svg(path, ...)` takes the same and writes the file. `Interface.render` and `Comparison.render` take these options too.

**How the residues and contacts are drawn**

| option | default | |
|---|---|---|
| `residues` | `None` | "atoms", "bubbles" or "names", or residue by residue: {"Gln712": "atoms", "*": "bubbles"}; None: the style's |
| `lines` | `None` | the colours and dashes of the contact lines: "detailed" or "schematic"; None: the style's |
| `lining` | `None` | the wall of the pocket: "contacts", "nearest" or "none"; None: the style's |
| `exposure` | `None` | grey halos on the ligand's atoms that are open to solvent; None: the style's |
| `charge_halos` | `False` | shade the charged group that a salt bridge or a pi-cation contact is made with |
| `charge_key` | `True` | name that shading in the key |
| `break_lines` | `True` | break a line where it runs over a word that is not its own |
| `atom_labels` | `'elements'` | what is written on an atom of a residue: "elements" (N, O), or "names", as in the file (NZ, OE1) |
| `distance_labels` | `'auto'` | the lengths written on lines: "auto" (hydrogen and halogen bonds, metals), "all" or "none" |
| `show_chain` | `'auto'` | name residues with their chain: "auto" (where there are several), "always" or "never" |
| `interactive` | `False` | the figure answers the pointer in a browser, and data-active on its root (a style sheet, no script) |
| `stylesheet` | `'inline'` | in a page with several figures: "inline", "scoped" (colours for this one alone) or "external" (result.css) |

**Compared with another protein**

| option | default | |
|---|---|---|
| `absent` | `None` | the residues the other one does not have (numbers or names); those in contact are marked and counted |
| `absent_label` | `None` | its name, for the key and the sentence under it |
| `absent_style` | `'marked'` | "marked" (a stroke of a marker pen) or "faint" |

**Colours and words**

| option | default | |
|---|---|---|
| `colors` | `'standard'` | the colour scheme: "standard", or "contrast"; the theme picks its light or dark palette |
| `background` | `True` | False: no background, transparent |
| `title` | `None` | a title above the figure |
| `subtitle` | `None` | a line under the title |
| `legend` | `True` | the key |
| `caption` | `True` | the sentences under the key |
| `notes` | `True` | of those, the ones on the ligand's own quirks |
| `bare` | `False` | the drawing and its key and no words; they are in result.notes |

**The confidence of a predicted model**

| option | default | |
|---|---|---|
| `confidence` | `True` | draw the model's confidence, where the diagram has it |
| `plddt_colors` | `False` | pLDDT in AlphaFold's colours and not by weight: True (behind the atoms) or "bonds" |
| `plddt_by_atom` | `True` | each atom drawn of a residue at its own pLDDT, where the model gives one for each |
| `pae_colors` | `None` | the predicted aligned error of a contact as a glow behind its line; None: for a ligand, not for an interface |

**The pocket and an ensemble**

| option | default | |
|---|---|---|
| `burial` | `True` | the ligand's buried fraction under the title |
| `solvent_sites` | `False` | predict waters that could bridge ligand and protein, and draw them, marked as predicted |
| `solvent_mark` | `'glow'` | how a predicted water stands on its line: "glow", "filled" or "outline" |
| `occupancy_style` | `'steps'` | the share of an ensemble: "steps" (three weights, with the number) or "continuous" (weight alone) |
| `occupancy_labels` | `'all'` | where the number stands: "all" (every contact) or "partial" (lines that fewer than all models make) |

**Size and legibility**

| option | default | |
|---|---|---|
| `figure_width_mm` | `180.0` | the width of the page the figure is for |
| `figure_height_mm` | `240.0` | its height |
| `min_font_pt` | `5.0` | the smallest text that counts as legible on it |
| `on_unreadable` | `'summary'` | a figure that is not legible: "summary" (a table), "render" (drawn all the same) or "none" |
| `effort` | `'full'` | for a peptide drawn as a chain: "full", or "quick" (one that is far too crowded is given up early) |
| `bond_length` | `None` | the scale, in page units for a bond; None: the largest that fits the page |
| `coords` | `'auto'` | how the ligand is laid out in two dimensions: "auto" (CoordGen unless its atoms clash), "coordgen" or "rdkit" |

## A ligand in its pocket

### `Diagram`

A ligand, its named atoms, and the interactions resolved onto them.

| | |
|---|---|
| `Diagram.from_structure(structure, ligand, smiles=None, interactions=None, *, name=None, ionize_acids=True, plddt='auto', model_confidence='auto', trusted=None, flag=None, ensemble=None, ligand_as='auto', detail=None, covalent=False, pae=None, ions=False)` | Build a diagram from a structure file. |
| `.n_interactions` | How many contacts the diagram has. |
| `.fingerprint(name=None)` | The per-(residue, interaction type) fingerprint of this complex (see `bindview.fingerprint`). |
| `.unresolved` | The contacts that could not be attached to the ligand: they are not drawn, and the result says so. |
| `.ligand_exposure()` | Relative solvent exposure (0..1) of every ligand atom, or `None` without a structure. |
| `.burial()` | How much of the ligand's surface the rest of the model covers. |
| `.solvent_sites()` | Predicted solvent sites that could bridge ligand and protein, best first. |
| `.proximal_residues(cutoff=4.5)` | Labels of residues with a heavy atom within `cutoff` angstrom of the ligand. |
| `.pocket_residues(cutoff=4.5)` | Residues with a heavy atom within `cutoff` angstrom of the ligand, in chain/number order. |
| `.render(style='detailed', theme='light', *, metrics=None, graph=None, **options)` | Lay out and draw the diagram. |
| `.render_svg(path=None, style='detailed', theme='light', **options)` | Render and, when `path` is given and there is a figure, write it. |
| `.arrange(path=None, style='detailed', *, title=None, port=0, browser=True, **options)` | Arrange the figure by hand, in a browser (`bindview.arrange`). |

### `RenderResult`

RenderResult(status: 'str', svg: 'str | None', figure: 'str', warnings: 'list[str]' = <factory>, quality: 'Quality | None' = None, path: 'Path | None' = None, style: 'str' = '', theme: 'str' = '', scene: 'Scene | None' = None, atoms: 'dict[str, tuple[float, float]]' = <factory>, confidence: 'dict[str, Any] | None' = None, data: 'dict[str, Any]' = <factory>, graph: 'Graph | None' = None, burial: 'dict[str, Any] | None' = None, notes: 'list[str]' = <factory>, css: 'str | None' = None)

Fields: `status`, `svg`, `figure`, `warnings`, `quality`, `path`, `style`, `theme`, `scene`, `atoms`, `confidence`, `data`, `graph`, `burial`, `notes`, `css`.

| | |
|---|---|
| `.ok` | Is it the diagram that was asked for, with nothing left out or assumed (status `ok`)? Not so where the figure is drawn and says that something is missing, nor where another figure or none was drawn. |
| `.is_empty` | Is it the figure of a complex in which no contact was found? |

### `Quality`

Measured properties of a laid-out diagram.

Fields: `width`, `height`, `bond_length`, `n_residues`, `n_interactions`, `overlaps`, `unplaced`, `line_crossings`, `bond_crossings`, `max_stretch`, `min_font_pt`, `required_width_mm`, `readable`, `ligand_width`, `estimated`, `link_hits`, `ligand_height`, `faithful`.

| | |
|---|---|
| `.valid` | Does nothing overlap, and was every residue placed? |

### `stylesheet(figures)`

One style sheet for a page with several figures (`render(stylesheet="external")`): what each of them goes with, every rule once. `figures`: their results, or their sheets (`result.css`).

## Contacts

### `Interaction`

One protein-ligand interaction.

Fields: `type`, `residue_number`, `residue_name`, `chain`, `protein_atom`, `ligand_atom`, `distance`, `angle`, `ligand_is_donor`, `beyond_contact_set`, `ligand_atoms`, `protein_atoms`, `insertion_code`, `water`, `occupancy`, `confidence`, `predicted`, `pae`, `pae_draws`.

| | |
|---|---|
| `Interaction.from_mapping(row)` | Build a record from a dict-like row (a TSV line, a database row). |
| `.with_(**changes)` | The same record with some of its fields changed. |
| `.residue_label` | `Gln712` -- three-letter name in title case plus number. |
| `.residue_id` | `(chain, number, insertion code)` of the residue. |

### `InteractionType`

The interaction classes the diagram knows how to draw.

| | |
|---|---|
| `.label` | The kind in words, as the key of a figure has it. |

### `load_interactions(path)`

Read an interaction table (`.tsv` / `.csv`) from disk.

### `parse_interactions(text, delimiter=None)`

Parse delimited text with a header row.

### `detect_interactions(structure, ligand)`

Detect interactions with PLIP (optional, GPL-2.0). See `bindview.detect`.

## Structures

### `Structure`

A parsed structure: a flat atom list plus lookup tables.

Fields: `atoms`, `source`, `chem_comp_smiles`, `plddt_scale`, `qa_metrics`, `links`, `model`, `accessions`.

| | |
|---|---|
| `.coords` | `(n_atoms, 3)` array of coordinates. |
| `.residues()` | All residues, in file order. |
| `.residue_atoms(key)` | The atoms of a residue. |
| `.residue_name(key)` | The name of a residue, as the file gives it. |
| `.find_residue(chain, number, icode='')` | Locate a residue by number; `chain=None` matches any chain if unique. |
| `.declares_plddt` | Does the file itself say that its B-factor column is a per-atom pLDDT? |
| `.chains_of(accession)` | The chains that the file assigns to an entry of a sequence database, by their author names, in the order of the file: a UniProt accession (`"P69905"`), or that of an isoform (`"P27348-2"`) where the file names the isoform. |
| `.chain_of(accession)` | The one chain that the file assigns to an entry of a sequence database (`chains_of`): for a caller who knows a protein by its accession and not by the name of its chain in this file. |
| `.polymer_residues(chain)` | The residues of `chain` that make up its polymer, in file order. |
| `.split(chain, parts)` | The structure with stretches of `chain` under chain names of their own. |
| `.with_chain_as_ligand(chain)` | The structure with the polymer of `chain` turned into one ligand residue. |
| `.with_residues_as_ligand(keys)` | The structure with these residues of one chain turned into one ligand residue. |
| `.resolve_ligand(spec)` | Resolve a ligand selection that may also name a whole polymer chain. |
| `.atom_by_name(key, name)` | An atom of a residue by its name, or None. |
| `.select_ligand(spec)` | Resolve a ligand selection to exactly one residue. |

### `read_structure(path, model=None)`

Read an mmCIF (`.cif`/`.mmcif`) or PDB (`.pdb`/`.ent`) file, gzipped (`.gz`) or not.

### `read_models(path)`

Every model of a structure file as a structure of its own, in the order of the file.

## The layout

### `Graph`

A diagram as nodes, contacts and rigid groups.

Fields: `view`, `bond_length`, `nodes`, `contacts`, `groups`, `origin`, `size`, `ensemble`, `springs`, `panels`, `meta`.

| | |
|---|---|
| `.find(ident)` | The node with exactly this id, or None. |
| `.node(ident)` | The node with this id, or the residue that the figure calls so: `"R:A:MET743"`, `"Met743"`, `"A:Met743"`, `"743"`; and `"ligand"` for any atom of it (it is one body). |
| `.contact(ident)` | The contact with this id. |
| `.group(ident)` | The rigid group with this id. |
| `.of_kind(kind)` | All nodes of a kind: `"residue"`, `"ligand_atom"`, ... |
| `.page(node)` | Where a node is on the page of the SVG (`None` if it was not placed). |
| `.move(ident, dx, dy)` | Move a node the way a drag does, and return the ids of what moved. |
| `.rotate(ident, degrees)` | Turn a node about itself, clockwise on the page, and return the ids of what turned. |
| `.flip(ident)` | Mirror a residue that is drawn with atoms in its own axis, as a page is turned over. (A ligand is not flipped: its drawing shows its stereochemistry.) |
| `.legend_offset()` | How far the legend was moved from where the page puts it; `(0, 0)` for a graph without one. |
| `.validate()` | Raise `GraphError` unless ids are unique, every reference resolves, and the scale and every place are numbers that a figure can be drawn from. |
| `.to_dict()` | The graph as plain data (`docs/graph-schema.md`). |
| `.to_json(path=None, indent=1)` | The graph as JSON text (key order fixed, floats in full precision, so the same graph always gives the same bytes). Written to `path` when given; always returned. |
| `Graph.from_dict(data)` | A graph from plain data; refused where it is not one. |
| `Graph.from_json(source)` | Read a graph from a file or from JSON text. A `Path` is a file; a string is one where it is the name of a file that is there, and JSON text where it is not. |

## Several models, several ligands

### `Ensemble`

Several models of one complex, each built as a `bindview.diagram.Diagram`.

| | |
|---|---|
| `Ensemble.from_structures(models, ligand, smiles=None, interactions=None, *, names=None, member=None, ensemble=None, **options)` | Build an ensemble from model files and one contact table for each. |
| `.n` | How many models the ensemble has. |
| `.table` | The models as fingerprints, one row a model: per residue and interaction type, as `Comparison.table` is for ligands. `table.consensus()` gives, per residue, the share of the models that contact it at all. |
| `.to_csv(path=None)` | The contacts as a table, one row each (`CONTACT_COLUMNS`); written to `path` if given. |
| `.diagram(absent='named')` | The drawn member as a single diagram whose contacts carry the ensemble as their occupancy. |
| `.residues` | Every residue that any model contacts, with the models that do: the most frequent first (`bindview.consensus.ResidueShare`). What `consensus` draws. |
| `.consensus(least=0.25, atoms=0.5)` | All the models in one diagram, counted by residue (`bindview.consensus`). |

### `EnsembleContact`

One contact of an ensemble: which models make it, and at what distance.

Fields: `chain`, `residue_number`, `insertion_code`, `residue_name`, `type`, `ligand_atoms`, `models`, `n_total`, `in_member`, `distances`.

| | |
|---|---|
| `.n_models` | How many models make it. |
| `.occupancy` | The share of the models that make the contact. |
| `.distance_median` | The median of its distance over the models that make it. |
| `.distance_min` | The shortest of its distances. |
| `.distance_max` | The longest of its distances. |
| `.residue_id` | `(chain, number, insertion code)` of the residue. |
| `.residue_label` | The residue as a figure names it. |
| `.as_dict()` | The contact as plain data. |

### `ResidueShare`

A residue, and the models of an ensemble that contact it.

Fields: `chain`, `residue_number`, `insertion_code`, `residue_name`, `models`, `n_total`, `kinds`, `places`.

| | |
|---|---|
| `.n_models` | How many models contact the residue. |
| `.share` | The share of the models that do. |
| `.residue_id` | `(chain, number, insertion code)` of the residue. |
| `.residue_label` | The residue as a figure names it. |
| `.said` | In words, for a tooltip. |
| `.models_at(part)` | The models that have it on one part of the ligand. |
| `.kind_at(part)` | The kind of contact that most models make with this part of the ligand. |

### `Comparison`

Two or more ligands of one pocket, each built as a `bindview.diagram.Diagram`.

| | |
|---|---|
| `.n` | How many ligands are compared. |
| `.consensus()` | Per residue, how many ligands contact it and how (see `FingerprintTable.consensus`). |
| `.consensus_contacts()` | The set as contact records, one per (residue, interaction type): `occupancy` is the share of the set's ligands that make the contact (see `FingerprintTable.contacts`). |
| `.diagram(k=0)` | Ligand `k` as a single diagram whose contacts carry the series as their occupancy. |
| `.difference(a=0, b=1)` | Shared / only-A / only-B contacts of two ligands of the set. |
| `.representative` | The ligand closest to the common substructure (fewest atoms outside it), when there is one. |
| `.core_atoms(k)` | Names of ligand `k`'s atoms in the common substructure (empty when it is not aligned on). |
| `.contact_via()` | Per residue: `(ligands that contact it through an atom of the common core, ligands that contact it only through atoms outside the core)`. Empty unless the set is aligned. |
| `.without_bond_orders` | The ligands of which only the connectivity is known (no SMILES, and no hydrogens to infer the bond orders from). The substructure search takes each of their bonds for a single one, so a double bond of another ligand has no match in them: the common substructure can be smaller for it, or missing. |
| `.alignment()` | The alignment decision as plain data. |
| `.render(view='auto', theme='light', *, metrics=None, pair=(0, 1), min_ligands=1, thumbnails=True, **options)` | Draw the comparison. |
| `.render_svg(path=None, view='auto', theme='light', **options)` | Render and, when `path` is given and there is a figure, write it. |

### `Difference`

Contacts of two ligands, classified. A contact is a (residue, interaction type) pair, and a residue is a chain, a number and an insertion code, whatever the two structures call it: a shared contact carries the name that the first ligand's structure gives its residue.

Fields: `names`, `shared`, `only_a`, `only_b`.

| | |
|---|---|
| `.tanimoto` | How alike the two are: shared bits over all bits. |
| `.residues()` | Each contacted residue as `"shared"`, `"a"` or `"b"` (contacted by both ligands in any way, or by one of them only). |
| `.as_dict()` | The classification as plain data. |

### `Fingerprint`

The set bits of one ligand, with what is known about each.

Fields: `name`, `bits`.

| | |
|---|---|
| `Fingerprint.from_diagram(diagram, name=None)` | The fingerprint of a built `bindview.diagram.Diagram`. |
| `.columns` | The set bits, in the fixed order. |
| `.vector(columns=None, counts=False)` | The fingerprint over `columns` (default: its own bits): 1/0, or record counts. |
| `.residues()` | Interaction types per contacted residue, both in the fixed order. |
| `.tanimoto(other)` | Shared bits over all bits set in either fingerprint (1.0 for two empty ones). |
| `.difference(other)` | `(shared, only in self, only in other)`, each in the fixed order. |
| `.rows()` | One row for each bit, as plain data. |
| `.to_csv(path=None)` | Long-form CSV: one row per set bit. Written to `path` when given; always returned. |
| `Fingerprint.from_csv(source)` | Read one fingerprint back from `to_csv` output (text or a path). |

### `FingerprintTable`

The fingerprints of a set of ligands over one residue numbering.

Fields: `fingerprints`.

| | |
|---|---|
| `.names` | The names of the ligands, in order. |
| `.columns` | The union of all set bits, in the fixed order. |
| `.matrix(counts=False)` | The bits as a matrix, one row a ligand; with `counts` the number of records. |
| `.consensus()` | Per residue: how many ligands contact it and with which types. |
| `.contacts()` | The set as contact records: one per (residue, interaction type), in the fixed column order, with `occupancy` the share of ligands that make the contact. |
| `.to_csv(path=None, form='long', counts=False)` | `form="long"`: one row per (ligand, set bit), lossless. `form="wide"`: one row per ligand and one column per bit of the union, holding 1/0 (or record counts). |
| `FingerprintTable.from_csv(source)` | Read either CSV form back (text or a path). The wide form carries presence (or counts) only; distances, confidence and atoms are in the long form. |

## Chains

### `Interface`

The contacts between two (or three) polymer participants as one figure.

| | |
|---|---|
| `Interface.from_structure(structure, participants, **options)` | The same as the class itself, under the name the other figures are built by (`Diagram.from_structure`, `Ensemble.from_structures`). |
| `.reach(column)` | How far the contacts of the residues around a column are spread along it, summed over those residues (in boxes). It is what makes lines long, makes them cross, and makes a residue be drawn twice; of two orders of the column the one with the smaller reach is the better drawing. |
| `.others` | The participants that stand around the spine. |
| `.title` | What the figure is called unless it is given a title. |
| `.render(style='detailed', **options)` | Render the figure. When the site that was chosen cannot be drawn legibly and nobody asked for it by name, the figure narrows to a smaller one, twice at most, and says so. |
| `.render_svg(path=None, style='detailed', **options)` | Draw the interface and write it to `path`, where one is given. |

### `Participant`

One side of an interface: a name, and the chains it consists of.

Fields: `name`, `chains`, `detail`.

| | |
|---|---|
| `Participant.parse(spec)` | `"A"`, `"D+E"` or `"receptor=D+E"`; or a sequence of chain names. |

### `InterfaceContact`

One contact across an interface, with the residue and the atoms on both sides.

Fields: `type`, `chain_a`, `number_a`, `resname_a`, `atoms_a`, `chain_b`, `number_b`, `resname_b`, `atoms_b`, `icode_a`, `icode_b`, `distance`, `angle`, `a_is_donor`, `confidence`, `pae`, `water`, `pae_draws`.

| | |
|---|---|
| `.residue_a` | `(chain, number, insertion code)` on the first side. |
| `.residue_b` | `(chain, number, insertion code)` on the second side. |
| `.sort_key` | The order of contacts in a table. |
| `.flipped()` | The same contact seen from the other side. |
| `.as_interaction()` | The contact as a record of a ligand diagram whose ligand is side `a` (the spine): its atoms are named as in a chain ligand, `63/OE1`. |

### `load_interface_contacts(path)`

Read a table written by `dump_interface_contacts` (or by hand, with those columns). Empty lines and lines that begin with `#` are passed over; an error names the line of the file.

### `BasePairs`

Two chains of RNA or DNA and the base pairs between them, as their secondary structures facing each other (see the module's words).

| | |
|---|---|
| `.across` | The pairs from one strand to the other. |
| `.within` | The pairs within a strand. |
| `.source_line()` | Where the pairs are from, in a sentence. |
| `.render(theme='light', *, metrics=None, title=None, legend=True, caption=True, background=True, colors='standard', step=24.0, figure_width_mm=180.0, figure_height_mm=240.0, min_font_pt=5.0, interactive=False, graph=None)` | Draw the figure. `theme`: `"light"`, `"dark"`, `"auto"` (both, following the reader) or a `bindview.Palette`; `colors`: the colour scheme. `step`: the distance of two neighbours of a strand, in px. `legend`: the key under the drawing; `caption`: the line of the source, which the result keeps in its `notes` also where it is left off the figure. |
| `.render_svg(path=None, theme='light', **options)` | Render and, when `path` is given, write the figure there. |

### `BasePair`

Two nucleotides whose bases pair, as a source says.

Fields: `chain_a`, `number_a`, `chain_b`, `number_b`, `family`, `icode_a`, `icode_b`, `ordered`, `symmetry`, `source`.

| | |
|---|---|
| `.key_a` | `(chain, number, insertion code)` of the first nucleotide. |
| `.key_b` | `(chain, number, insertion code)` of the second. |

### `BaseStack`

Two bases stacked on each other, as a source says.

Fields: `chain_a`, `number_a`, `chain_b`, `number_b`, `icode_a`, `icode_b`, `source`.

| | |
|---|---|
| `.key_a` | `(chain, number, insertion code)` of the first nucleotide. |
| `.key_b` | `(chain, number, insertion code)` of the second. |

### `read_mmcif_pairs(path_or_text)`

The base pairs the PDB annotated in a deposited entry: `_ndb_struct_na_base_pair` of its mmCIF file (a path, gzipped or not, or the text). Chains and numbers are the authors', and the label ids where the table has no others, as the atoms of such a file are named. Of an entry of several models, the pairs of the first that the table lists. A pair whose partner is a crystal mate keeps its operator in `BasePair.symmetry`. `hbond_type_12` gives the family; for the asymmetric ones (cWH, tWH, cWS, tWS, cHS, tHS) the file does not say which base uses which edge, and the pair says so (`ordered=False`). A file without the category gives no pairs.

### `read_fr3d_pairs(path_or_text, source='fr3d')`

The base pairs of an FR3D table as RNA 3D Hub serves it (CSV: unit id, family, unit id; a path or the text). Each pair is listed once though the table lists it from both sides; near pairs (`ncWW`) are left out. `source` is what the figure credits: the default credits RNA 3D Hub; give your own words for a table made elsewhere.

### `read_fr3d_stacks(path_or_text, source='fr3d')`

The stacks of RNA 3D Hub's FR3D stacking table (`s33`, `s35`, `s53`, `s55`), each once. A stack with a crystal mate (a unit id with a symmetry operator other than the identity, `1_555`) is left out: a figure has the chains of the file.

### `read_rnapolis_pairs(path_or_json, source='rnapolis')`

The base pairs of the JSON that RNApolis's `annotator` writes (`annotator -j pairs.json model.pdb`), a path, its text or the parsed object. The family (`lw`) names the first nucleotide's edge first; a pair without one is read without a family. Each pair once.

### `read_rnapolis_stacks(path_or_json, source='rnapolis')`

The stacks of RNApolis's JSON (`stackings`), each once; a figure draws those across its strands.

### `pairs_from_dot_bracket(structure, chain, dot_bracket, family=None, source='dot-bracket')`

The pairs of one strand's secondary structure written as a dot-bracket, one character a nucleotide of `chain` in the order of the chain: `()`, `[]`, `{}`, `<>` for pairs, a dot (or any other sign) for an unpaired nucleotide. A letter is refused: some notations write a further level of pseudoknots with letters (`A` closing with `a`), which are not read here and would be pairs lost without a word. No family unless the caller names one (`family="cWW"` for a secondary structure of Watson–Crick pairs).

### `pairs_from_contacts(contacts)`

A stand-in where no annotation is at hand: from bindview's plain distance table between two chains (`bindview.distance_table.interface_contacts`, with `stacking=True` for the stacks), two nucleotides with two or more hydrogen bonds from base to base are a pair *without a family*, unless the table also finds them stacked (a pair and a stack exclude each other); the stacks of the table are stacks. A row may be written from either side: a bond is counted once, and a pair or a stack is named from the side its first row is written from. Nothing here classifies a pair.

### `pairs_within_from_table(structure, chain)`

The stand-in of `pairs_from_contacts` within one strand, where no annotation and no dot-bracket is at hand: two nucleotides of the strand that are not neighbours, with two or more hydrogen bonds from base to base by the plain distance table's rule (an N or O within `bindview.distance_table.HBOND` of an N or O, one able to donate and the other to accept), are a pair *without a family*. A nucleotide is in one pair at most: pairs are taken most bonds first, then the shorter mean bond, and one whose nucleotide is taken is left out -- so a base triple shows as one pair. Stacks within a strand are not looked for; nothing here classifies a pair.

### `Peptide`

Peptide(residues: 'tuple[PeptideResidue, ...]', appendages: 'tuple[Appendage, ...]' = (), crosslinks: 'tuple[Crosslink, ...]' = (), cyclic: 'bool' = False, runs: 'tuple[int, ...]' = (), n_atoms: 'int' = 0, numbering: 'str' = 'derived', breaks: 'tuple[int, ...]' = (), open_ends: 'tuple[bool, bool]' = (False, False), pairwise: 'bool' = False, called: 'str' = 'the ligand', sequential: 'bool' = True, detail: "'str | None'" = None, _owner: 'dict[int, int]' = <factory>)

Fields: `residues`, `appendages`, `crosslinks`, `cyclic`, `runs`, `n_atoms`, `numbering`, `breaks`, `open_ends`, `pairwise`, `called`, `sequential`, `detail`.

| | |
|---|---|
| `.residue(position)` | The residue at a position, counted from 1. |
| `.residue_of(atom)` | The residue an atom belongs to, or `None` for an atom of an appendage. |
| `.appendage_of(atom)` | What is bonded to the chain and is no residue of it, at this atom. |
| `.sequence` | One-letter codes from the N-terminus; `X` for what is not one of the twenty. |
| `.recognised` | The residues whose side chain is known. |
| `.in_chain` | Heavy atoms that belong to a residue of the chain. |

### `perceive_peptide(mol, stereo_from=None)`

The peptide chain in a molecule, or `None` if it has no two consecutive backbone units.

## Confidence and colours

### `Confidence`

Per-residue and per-ligand-atom pLDDT of one model, on the 0-100 scale.

Fields: `residues`, `ligand_atoms`, `pocket_plddt`, `pocket_residues`, `origin`, `model`, `given_on`, `atoms`, `per_atom`.

| | |
|---|---|
| `Confidence.from_structure(structure, ligand, *, scale=1.0, origin='declared', cutoff=4.5, model=None)` | Read pLDDT from the B-factor column of `structure`. |
| `Confidence.from_values(residues, ligand_atoms=None, *, pocket_plddt=None, model=None, scale=None)` | Confidence from values the caller already has. |
| `.residue(key)` | pLDDT of a residue: the mean over its atom records. |
| `.atom(key, name)` | pLDDT of one atom of a residue: its own where the model gives the atoms of a residue values of their own and this one has one, the residue's otherwise. |
| `.ligand(names)` | pLDDT of a ligand site: the atom's own value, or the mean over the atoms of a ring or charged group (whose centre is what the contact attaches to). |
| `.contact(residue, names)` | Confidence of a contact: the **lower** of its two ends. |
| `.ligand_plddt` | Mean pLDDT over the ligand's atoms. |
| `.marked` | Has the caller flagged this model as not to be trusted? |
| `.with_model(model)` | The same with the numbers of the model as a whole. |
| `.summary()` | The figure-level numbers, as plain data. |

### `ModelConfidence`

Model-level evidence, supplied by the caller and displayed as given.

Fields: `iptm`, `ptm`, `ranking_score`, `trusted`, `flag`.

| | |
|---|---|
| `ModelConfidence.from_mapping(data, **overrides)` | From a mapping, such as AlphaFold 3's summary: `iptm`, `ptm` and `ranking_score` as numbers, or as text that is one (a table gives `"0.83"`); `trusted` as true or false, or as one of those words; `flag`. A key that is not there, or holds nothing, is left out. A value that is neither is refused by its key, not read as something else. |
| `ModelConfidence.from_json(path, **overrides)` | Read an AlphaFold 3 `summary_confidences.json` (or any JSON with those keys). |
| `ModelConfidence.beside(structure_path)` | The summary file AlphaFold 3 writes next to a model, if it is there. |
| `ModelConfidence.coerce(value)` | From whatever `model_confidence` was given. |

### `Palette`

Named colours. Tokens are the attribute names; element and residue-class colours live in the two mappings.

Fields: `name`, `background`, `text`, `text_muted`, `ligand_bond`, `ligand_carbon`, `protein_bond`, `protein_label`, `protein_atom`, `hbond`, `hydrophobic`, `saltbridge`, `pi_stacking`, `pi_cation`, `halogen`, `metal`, `water_bridge`, `coordination`, `beyond`, `exposure`, `warning`, `panel`, `panel_border`, `sch_hbond`, `sch_saltbridge`, `sch_pi_stacking`, `sch_pi_cation`, `sch_halogen`, `sch_metal`, `sch_water_bridge`, `bubble_text`, `bubble_line`, `elements`, `residue_classes`, `extra`.

| | |
|---|---|
| `.element(symbol, on_protein=False)` | The colour of an element's letter: of a carbon, the colour of the bonds it is drawn with. |
| `.interaction(kind)` | The colour of a kind of contact, by its name in a table (`"hbond"`, `"pi_stacking"`). |
| `.element_token(symbol, on_protein=False)` | The colour *token* for an element label (see `tokens`). |
| `.residue_class(cls)` | The colour of a class of residues. |
| `.with_(**changes)` | The same palette with some of its colours changed. |
| `.tokens()` | Flat `token -> colour` map (used for CSS-variable output). |

### `Metrics`

Dimensions, in SVG user units (px at natural size). Everything scales from `bond_length`; fonts are absolute because legibility is.

Fields: `bond_length`, `bond_width`, `double_offset`, `wedge_width`, `hash_spacing`, `label_gap`, `atom_font`, `sub_font`, `charge_font`, `protein_bond_width`, `protein_bond_scale`, `protein_atom_font`, `residue_font`, `interaction_width`, `distance_font`, `legend_font`, `title_font`, `caption_font`, `lash_length`, `spoke_inner`, `spoke_outer`, `margin`, `chain_pitch`.

| | |
|---|---|
| `.with_(**changes)` | The same dimensions with some of them changed. |

## Errors

### `BindviewError`

Base class for every error raised by this package.

### `StructureError`

The structure file could not be read or contains no usable atoms.

### `LigandNotFoundError`

The ligand selection matched nothing (or matched ambiguously).

### `LigandBuildError`

An RDKit molecule could not be built for the ligand.

### `InteractionError`

An interaction record is malformed.

### `DetectionUnavailableError`

Interaction detection was requested but the optional detector is not installed.

### `RenderError`

The diagram could not be rendered.

### `GraphError`

A diagram graph is malformed, of an unsupported version, or does not fit the diagram.

### `EnsembleError`

Models given as one ensemble are not models of one complex: another ligand, or another pocket.

## Functions of modules

### `bindview.distance_table.ligand_contacts(structure, ligand, smiles=None, ionize_acids=True)`

The table for a ligand that is one residue; rows in a fixed order, ready for bindview.

### `bindview.distance_table.contact_table(structure, chain, partners=None, *, stacking=False)`

The table for the polymer of `chain` as ligand; rows in a fixed order, ready for bindview.

### `bindview.distance_table.interface_contacts(structure, chains_a, chains_b, *, stacking=False)`

The same plain table for an interface between two sets of chains, as `bindview.interface.InterfaceContact` records with side `a` in `chains_a`. `stacking` as for `contact_table`.

### `bindview.arrange.page(diagram, path, title=None, **options)`

Write the page as one file, with a preview and without bindview behind it. `options`: those of `render`.

### `bindview.arrange.serve(diagram, title=None, *, port=0, browser=True, **options)`

Open the page with bindview behind it, and hand back the layout when it is done (None if it is stopped with Ctrl-C). `options`: those of `render`.

### `bindview.report.survey(path, smiles=None, domains=None)`

Look at a model, or at a folder that holds one or a set of them: what it holds, and the figures that follow.
