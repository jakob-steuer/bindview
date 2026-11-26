# The diagram graph: `bindview.graph`, schema versions 1.0 to 1.4

A bindview figure exists in two forms. The **scene** is a tree of drawing primitives and becomes the SVG. The
**graph** says what the figure *is*: nodes with an identity and a place, contacts between them, and groups of nodes
that move as one body. The layout engine's result is turned into a graph, and the renderers draw from the graph.

Serialised as JSON, the graph does three jobs with one format:

* the model a person or a program moves: every node says whether it may move, what holds it, and how strongly;
* the answer of a web service: identities, positions, contacts and their evidence, without parsing SVG;
* a regression target: it does not change when a colour, a stroke width or a font does.

```python
from bindview import Diagram, Graph, load_interactions

smiles = "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C"
dia = Diagram.from_structure("testdata/kit_imatinib/model.cif.gz", "STI", smiles=smiles, name="imatinib",
                             interactions=load_interactions("testdata/kit_imatinib/interactions.tsv"))
result = dia.render(style="detailed")
text = result.graph.to_json()           # fixed key order, full precision: the same graph gives the same text
graph = Graph.from_json(text)           # also from a path; to_json(path) writes one
graph.move("Thr670", 12, -6)            # a residue moves alone; "R:A:THR670" is its id
graph.move("L:N13", 30, 0)              # an atom of the ligand takes the whole depiction with it
again = dia.render(style="detailed", graph=graph, on_unreadable="render")
```

```bash
bindview render testdata/kit_imatinib/model.cif.gz --ligand STI --distances-only -o sti.svg --graph sti.json \
    --smiles "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C"
bindview render testdata/kit_imatinib/model.cif.gz --ligand STI --distances-only -o sti.svg --layout sti.json \
    --smiles "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C"
```

The first writes the figure and its graph; the second draws the figure from the graph, as it is or with nodes
moved.

`result.graph` is `None` for an empty figure and for a summary table. An untouched graph draws the identical figure,
byte for byte, and no layout is run for it. This page describes the JSON.

## Versions

Every document starts with `"schema": "bindview.graph"` and `"version": "major.minor"`.

* A reader accepts any document of its own **major** version and refuses others.
* A **minor** version only adds optional keys. A reader keeps the keys it does not know (`Graph.from_dict` stores
  them and `to_dict` writes them back), so a reader of 1.0 passes a 1.4 document through without loss.
* Removing or renaming a key, changing a meaning or a unit, and adding a node kind are major changes.
* A document has the lowest version that holds what it says:

| version | written where the graph has |
|---|---|
| `1.0` | none of the below |
| `1.1` | a ligand drawn as a chain of its residues: a `ligand_group` node with a `position`, or a contact with `ligand_residue` ([below](#a-ligand-drawn-as-a-chain-11)) |
| `1.2` | the figure of an interface (`meta.interface`), or a contact with a `pae` ([below](#an-interface-12)) |
| `1.3` | an interface whose boxes stand on a plane (`meta.interface.layout` is `"plane"`) |
| `1.4` | two strands and their base pairs (`meta.basepairs`) |

Keys that a figure at a level of `detail` adds (`beside`, `detail`, `plddt_basis`) and the `ipsae` of an interface
are optional keys of the version the document has otherwise. Absent optional keys mean "not applicable" or their
stated default; writers omit them.

## Coordinates

Positions are in SVG user units (px at the figure's natural size), y pointing down, in the **diagram's own frame**:
the frame the layout engine works in, with the ligand's drawing near the origin. `origin` is where that frame sits
on the page, so a node's position in the SVG is `(x + origin[0], y + origin[1])`; `size` is the page. Keeping the
two apart is what makes the round trip exact: positions are stored as computed. `bond_length` is the length of a
ligand bond in these units.

## Top level

| key | type | meaning |
|---|---|---|
| `schema`, `version` | string | `"bindview.graph"`, and `"1.0"` to `"1.4"` (above) |
| `view` | string | the renderer the graph was laid out for: `detailed`, `schematic`, `difference`, `consensus`, `basepairs` |
| `bond_length` | number | ligand bond length in user units |
| `origin` | [x, y] | offset of the diagram frame on the page |
| `size` | [w, h] | page size; absent before a page exists |
| `ensemble` | string | what the contacts' `occupancy` is a share of ("1,000 frames of a 100 ns trajectory", "the 10 ligands of the set"); absent where no contact carries one |
| `springs` | object | the layout engine's constants, see [Springs](#springs) |
| `panels` | array | difference view only: one entry per ligand, `{id, group, label, origin}`: the page offset of that ligand's panel |
| `meta` | object | facts about the figure (`ligand`, `ligands`, `n_interactions`, `frame`, `drawn_ligand`, …), and `burial`: the ligand's solvent-accessible surface free and in the complex (`sasa_free`, `sasa_complex`, Å²), its `buried_fraction` and `exposed_fraction`, the `probe` (Å) and `n_points` an atom, the cut `warn_below`, and `warning`: whether it fell below (absent for an interface, which has no ligand) |
| `groups` | array | rigid groups |
| `nodes` | array | nodes |
| `contacts` | array | contacts |

## Rigid groups

A group lists nodes whose positions relative to each other are fixed.

| key | type | meaning |
|---|---|---|
| `id` | string | `ligand`; `ligand:1`, `ligand:2` where a figure draws two; `ligand:at20` for what hangs off residue 20 of a ligand drawn as a chain |
| `kind` | string | `rigid`, the only kind in schema 1 |
| `label` | string | the ligand's name |
| `members` | [node id] | every atom, ring-centroid and group node of that ligand |
| `movable` | bool | whether the group may be dragged |
| `coordination` | [[metal, donor]] | metal–donor pairs drawn as coordination lines, by atom name; absent without a metal |
| `panel` | string | difference view: the panel the ligand is drawn in |

The 2D depiction of a ligand is one rigid body: its atoms have no freedom relative to each other, so **moving any
member moves all of them by the same vector**. The layout engine treats the ligand as the fixed frame that
everything else is placed around; it has no spring, because it is what the springs are attached to.

**A ligand drawn as a chain is not one rigid body.** Each of its residues is a node of its own and belongs to no
group, so it can be moved alone and the backbone follows. A residue that has something bonded to it outside the
chain (a group on a lysine, a cap) forms one rigid group with those atoms (`ligand:at<position>`). A chain does not
flex during layout.

## Nodes

| key | type | meaning |
|---|---|---|
| `id` | string | stable and unique within the document. Opaque: use `kind`, `group` and the attributes, do not parse it |
| `kind` | string | see the table below |
| `label` | string | what a person calls it: `C12`, `Gln570`, `ring` |
| `x`, `y` | number \| null | position in the diagram frame; `null` where the layout could not place the node |
| `angle`, `mirror` | number, bool | rotation (radians) and mirroring of the node's own drawing; default `0`, `false` |
| `group` | string | the rigid group the node belongs to; absent for a free node |
| `movable` | bool | whether it may be dragged |
| `anchors` | [node id] | the nodes it is tethered to |
| `spring` | number | summed weight of its tethers: how strongly it is held; absent where nothing holds it |
| `contacts` | [contact id] | the contacts that reference the node; for a residue, in tether order |
| `members` | [node id] | `ring_centroid`, `ligand_group`: the atoms the node stands for |
| `confidence` | number | pLDDT scale, 0–100: the atom's or the residue's own |
| `occupancy` | number | share of the ensemble, 0–1 (below: what it means for each kind) |

Any other key is an attribute of the node's kind.

| kind | what it is | attributes |
|---|---|---|
| `ligand_atom` | an atom of the drawn ligand | `element`; `exposure` (relative solvent exposure 0–1, where the style computed it); `drawing_only` (`true` for a hydrogen that exists only in the drawing, at a stereocentre) |
| `ring_centroid` | the centre of a ligand ring that a π interaction attaches to | (`members`) |
| `ligand_group` | a group of ligand atoms that a contact attaches to as a whole: a charged group, or a **residue of a ligand drawn as a chain** (1.1) | for a charged group: `members`, and `label` names it. For a residue of the ligand, `label` is its code and position (`Lys20`) and `members` is empty, because its atoms are not nodes of their own; instead `form` (`chain_residue`), `resname` (`LYS`; `XAA` where the side chain was not recognised), `position` (1-based, **counted from the N-terminus of the ligand's own chain, not a sequence-database number**), `number` (for a chain that the structure file gives as a chain of its own: the residue's number **as deposited**, as text, `63` or `63A`), `class`, `atoms` (the names of its atoms in the structure file), `backbone` (those of them that are backbone), `width`, `height` (of its box), and at a level of `detail` `beside` (the names of those of its `atoms` that stand beside its box, each of which is then a `ligand_atom` node in the residue's rigid group) |
| `residue` | a residue of the target | `resname`, `chain`, `number`, `icode`, `class` (hydrophobic, polar, positive, negative, glycine, nucleotide, metal, water, other), `form` (how it is drawn: `fragment`, `eyelash`, `chip`, `bubble`, `node`, `consensus`), `radius` (of a circle containing its drawing), `beyond` (`true` where all its contacts lie outside the caller's contact set). In a figure of all the models of an ensemble: `n_models` and `n_total`, how many of them contact it. Beside a ligand drawn as a chain, `drawing` and `drawings` where the residue is **drawn more than once**, beside each stretch of the chain it touches (its contacts lie more than 8 residues of the ligand apart); the `id` is then `R:<residue>~1`, `~2`, … |
| `legend` | the key | `width`, `height`, `offset` |
| `solvent_site` | a **predicted** water site between a ligand atom and a residue; emitted only where solvent sites were asked for, and never for a structure with waters of its own. It sits where its label is drawn on the bridge and is not movable by itself | `predicted` (always `true`), `score` (hydrogen-bond partners pointing at the site), `enclosure` (0 open solvent to 1 enclosed), `xyz` (the site in the structure's coordinates, ångström), `partners` (`protein` and `ligand` atoms with distances) |

A residue node's `occupancy` in a single-ligand figure is that of its most persistent contact: the level it is
drawn at, a lower bound on how often the residue makes any contact. In a consensus figure it is exact: the share of
the set's ligands that contact the residue in any way.

## Contacts

A contact is one interaction between a residue node and a ligand node. In the single and difference views it is also
a tether the layout engine placed the residue by.

| key | type | meaning |
|---|---|---|
| `id` | string | stable and unique; opaque |
| `type` | string | `hydrophobic`, `hbond`, `saltbridge`, `pi_stacking`, `pi_cation`, `halogen`, `metal`, `water_bridge`, `covalent`; in 1.4 also `base_pair` and `base_stack` |
| `residue` | node id | the target side |
| `ligand` | node id \| null | what it attaches to on the ligand: an atom, a ring centroid or a group. `null` in a consensus view, where a contact stands for a set of ligands |
| `drawn` | bool | whether it is a line. `false` for a contact without direction (hydrophobic), which is shown by proximity. Beside a ligand drawn as a chain every contact is a line, a hydrophobic one too |
| `distance` | number | ångström, as reported; in a consensus view the shortest over the set |
| `occupancy` | number | share of the ensemble in which **this contact** is made, 0–1 |
| `confidence` | number | how much the model is to be believed about **this contact**, pLDDT scale 0–100: the value the row carried, else the lower of the ligand site's pLDDT and the residue's |
| `pae` | number | (1.2) the predicted aligned error between the contact's two residues, ångström; read from a matrix, the smaller of its two directions. A claim about the pair, where `confidence` is about each end: the two are never combined. Not part of `label` |
| `pae_draws` | list of numbers | where `pae` is the median over several draws of a model: each draw's value, in the order the draws were given. Absent for a contact with one value |
| `label` | string | text printed on the line (a distance, `H₂O`, the low-confidence mark) |
| `label_t` | number | where along the line the label sits, 0 = residue end, 1 = ligand end |
| `ideal`, `min_length`, `weight` | number | the tether: rest length, shortest allowed length, spring weight |
| `site` | string | what makes it on the residue: `atom`, `ring`, `group`, `residue` |
| `protein_atoms` | [string] | atom names of that site |
| `records` | [int] | indices of the interaction records the contact stands for (a style may merge several) |
| `ligand_residue` | string | (1.1) the residue of a ligand drawn as a chain that makes the contact, as `Lys20`; `ligand` is then that residue's node |
| `ligand_atoms`, `ligand_part` | [string], string | with `ligand_residue`: the atoms of that residue that make the contact, and whether they are its `backbone`, its `side chain` (an amino acid) or its `base` (a nucleotide) |
| `beyond` | bool | outside the caller's contact set |
| `ligand_group`, `shared` | string, bool | difference view: which ligand the contact belongs to, and whether the other ligand makes the same contact |
| `predicted`, `via` | bool, node id | a water bridge through a predicted solvent site, and that site's node. Absent for anything observed or detected |

`occupancy` and `confidence` belong to the contact, not to the residue. They are separate numbers and are never
combined into one.

## Springs

`springs` carries the layout engine's own constants, not a second mechanism:

| key | meaning |
|---|---|
| `link` | weight of a tie that is drawn as a line |
| `sightline` | weight shared among the line-less contacts of a residue |
| `eyelash` | the same for a residue that makes line-less contacts only and is drawn as an eyelash (detailed style): its sightlines together pull like one link |
| `length` | the engine's `w_length` |
| `short` | the engine's `w_short`: how much stiffer a tie is in compression |
| `unit` | the length that makes the terms dimensionless: one bond length |

A tie of weight `w` and rest length `ideal` costs, at length `d`,

    length · w · ((d − ideal) / unit)²        if d ≥ ideal
    length · short · w · ((d − ideal) / unit)²  if d < ideal

and `d ≥ min_length` is a hard limit. Each contact carries its own `weight`, `ideal` and `min_length`; a node's
`spring` is the sum of its contacts' weights. So a residue is held by a finite spring and the ligand's internal
geometry by an infinitely stiff one: a dragged residue moves alone.

## Moving things

`Graph.move(id, dx, dy)` applies the two rules at the data level and returns what moved:

* a node of a rigid group moves the whole group;
* any other node moves alone;
* the legend moves by accumulating `offset`, relative to where the page places it.

`Graph.move` takes a node by its id or a residue by the name the figure gives it (`"Thr670"`, `"A:Thr670"`), and
`"ligand"` for the whole depiction. `Graph.rotate` turns a residue that is drawn with atoms (its `angle`), the
`"ligand"`, or the `"figure"` as a whole; `Graph.flip` turns a residue over (its `mirror`). `Diagram.arrange` does
the same by hand, in a browser.

Nothing else follows. Tethers are springs, not rods; whether to relax the layout after a move is the caller's
decision, and `springs` is what it needs to do so.

When a graph is handed back to `render(graph=…)`, these are **read**: `bond_length`; `x`, `y` of every ligand atom
(the depiction is drawn exactly there) and, for a ligand drawn as a chain, of every residue of it (the backbone is
drawn through them); `x`, `y`, `angle`, `mirror` of every residue node; `label_t` of every contact; the group's
`coordination`; the legend's `offset`. Everything else is computed again from the diagram and written fresh into
the graph of the figure that is drawn. A graph that does not fit the diagram (another ligand, other render options) is
refused with a `GraphError`; one laid out for the other style, with a `ValueError`.

A layout from outside is held to the same audit as one from the engine: where nodes overlap or a line runs through
another residue, the result is `unreadable` (`on_unreadable="render"` draws it anyway, while a person is still
moving things).

## Per view

**`detailed`, `schematic`**: one rigid group `ligand`, atom ids `L:<atom name>` (and `L:H@<atom name>` for a
drawing-only hydrogen), residue ids `R:<chain>:<RESNAME><number>`, one legend node. These graphs are drawn again
from.

**`difference`**: two rigid groups, `ligand:1` and `ligand:2`, both in the one diagram frame; `panels` gives the
page offset each is drawn at. Residue nodes exist once and are used by both panels: that is the invariant frame. A
residue's `contact` attribute says, per ligand, `shared`, `only` or `absent`; `confidence_by_ligand` holds its
pLDDT in each model.

**`consensus`**: residue nodes carry `occupancy`, `n_ligands`, `n_total`, `ligands`, `by_confidence` (contacting
ligands at pLDDT ≥ 70, 50–70, < 50) and, for an aligned set, `via_core` / `via_substituent_only`. Their contacts are
one per interaction type, with the share of ligands making it as `occupancy` and no ligand end. `meta.frame` says
which frame was drawn: `core` (a rigid group for the drawn ligand, residues anchored to its atoms), `envelope` (no
ligand; `anchor_point` on each residue) or `sequence` (no ligand, no anchors, no springs).

Comparison graphs are written and read back through JSON; `Comparison.render` does not take one, `Diagram.render`
does.

## A ligand drawn as a chain (1.1)

Each residue of the ligand is a `ligand_group` node with the attributes above, and `meta.chain` says how the ligand
is drawn: `drawn_as`, `why`, `residues`, `sequence`, `cyclic`, `numbering`, `numbering_origin` (`derived` for a
chain perceived from its bonds, `deposited` for one the structure file gives as a chain of its own; the node's
`number` is then the deposited one) and `atoms_outside_the_chain`. A chain of nucleotides as the ligand is written
the same way.

## An interface (1.2)

`meta.interface` holds the `participants` in order with their chains, the `spine`, the `spine_order` of the column
(`sequence` or `interface`; `null` where there is no column), where the contacts came from, and the `site` that is
shown: its `kind`, its `rule` in words, its `residues` of `of_residues` and its `contacts` of `of_contacts`. A residue of the spine is
a `ligand_group` node as in 1.1, and a residue around it a `residue` node. Every interface graph says `layout`, how
it came about in words (`layout_rule`; for a column drawn by default, what the rule would have chosen) and the
number of `stretches` of the chain its residues come in.

With a PAE, `meta.interface.pae` summarises it: `min`, `median`, `max` over the contacts shown, `with_a_value` of
`of_contacts`, `pairs` of `of_pairs` residue pairs with a value, `pairs_written` (pairs whose lines carry the tag),
the thresholds `written_from` and `bold_from`, and, where a matrix was read, `direction` (which of its two
directions a pair takes). For the values of several draws of a model: `draws` (how many), `over_draws` (`median`),
`by_draw` (each draw's `min`, `median` and `max` over the contacts shown), and for a model with copies of a chain
`copies` (the groups of chains that are copies of one another), `chains_matched_by` (`superposition`) and
`draws_named_otherwise` (for each draw whose chains were matched under other names: its place in the order given,
counted from 0, its file, which of its chains stands for which chain of the figure, and the CA RMSD with that
assignment and as named).

Where it was asked for, `meta.interface.ipsae` holds the ipSAE of the interface: the `cutoff` in ångström, how `d0`
is taken (in words), and `scores`, one entry for each direction of each pair of participants, with the participant
the model is `aligned_on`, the one it is `scored_over` and the `value` (0 to 1). For several draws the value is
their median, each entry has `by_draw`, and `draws` and `over_draws` say so.

**At a level of `detail`** (only where one was asked for): `beside` on the `ligand_group` nodes of the column;
`detail` under `meta` (the level of the `ligand` and of the `protein`) or under `meta.interface` (the level of each
participant, by name); and `plddt_basis` under `meta`, `"atom"` or `"residue"`: whether a drawn atom of a residue
is drawn at its own pLDDT or its residue's.

**On a plane (1.3)** the positions of the spine's `ligand_group` nodes mean something, and `plane` under
`meta.interface` says how much: `neighbours_kept` (the share of neighbour relations the figure has right), `false_neighbours` and
`lost_neighbours` of `neighbour_pairs`, `distance_error`, `px_per_angstrom` (the scale: a figure drawn again from
its graph is measured with it), `out_of_plane_rms` (ångström), `boxes_moved`, `partner_off_normal_deg`,
`boxes_covering_each_other` (only where it is not zero, which a layout never produces and a graph moved by hand
can), the same numbers for the bare `projection`, the `thresholds` a projection has to meet
(`neighbours_kept_min`, `distance_error_max`, `neighbour_within`, `apart_beyond`) and `"drawn": true`. Where a
plane was looked at and not drawn, a 1.2 document holds its numbers with `"drawn": false`.

## Base pairs (1.4)

The graph of `BasePairs` (`"view": "basepairs"`) has `meta.basepairs` with the `chains`, the `layout`, the `source`
line and the counts `pairs_across`, `pairs_within`, `stacks` and `hbonds`. Every nucleotide is a `residue` node (`chain`, `number`, `resname`, `letter`, `modified`,
`strand` 1 or 2, `paired_across`), and a ligand beside the pairs is a `ligand_group` node (`residue`,
`contacts_from`) without members. A contact joins two residue nodes, the second named by `ligand`: `base_pair`
(`family`, `ordered`, `across`, `pale`, `source`; `label` is the family), `base_stack` (`source`) and `hbond`
(another hydrogen bond across, with a sugar or phosphate atom: `bonds`, `with`). A contact of the ligand keeps its
own type, its `distance` and `protein_atoms` (the nucleotide's atom), with the ligand node as `ligand`. Positions are
those of the page (`origin` is `[0, 0]`); `BasePairs.render(graph=...)` draws a graph of the same figure again.

## Example

The hydrogen bond of Thr670 with N13 of imatinib, from the detailed figure of `testdata/kit_imatinib` in the
first block above (numbers rounded, lists shortened):

```json
{
 "schema": "bindview.graph", "version": "1.0", "view": "detailed", "bond_length": 30.0,
 "origin": [276.6, 183.0], "size": [663.6, 434.9],
 "springs": {"eyelash": 1.0, "length": 1.0, "link": 1.0, "short": 4.0, "sightline": 0.35, "unit": 30.0},
 "meta": {"burial": {"buried_fraction": 0.911, "warn_below": 0.7, "warning": false, "…": "…"}, "…": "…"},
 "groups": [{"id": "ligand", "kind": "rigid", "label": "imatinib", "members": ["L:C1", "L:C6", "…"], "movable": true}],
 "nodes": [
  {"id": "L:N13", "kind": "ligand_atom", "label": "N13", "x": 80.1, "y": 4.8, "group": "ligand",
   "movable": true, "contacts": ["A:THR670|hbond|L:N13|atom:OG1"], "element": "N"},
  {"id": "R:A:THR670", "kind": "residue", "label": "Thr670", "x": 12.1, "y": 42.4, "angle": 4.8, "mirror": true,
   "movable": true, "anchors": ["L:N13", "L:C19"], "spring": 1.35,
   "contacts": ["A:THR670|hbond|L:N13|atom:OG1", "A:THR670|hydrophobic|L:C19|atom:CG2"],
   "chain": "A", "class": "polar", "form": "fragment", "number": 670, "radius": 73.2, "resname": "THR"}
 ],
 "contacts": [
  {"id": "A:THR670|hbond|L:N13|atom:OG1", "type": "hbond", "residue": "R:A:THR670", "ligand": "L:N13",
   "drawn": true, "distance": 2.95, "label": "2.95", "label_t": 0.5, "ideal": 61.5, "min_length": 50.5,
   "weight": 1.0, "site": "atom", "protein_atoms": ["OG1"], "records": [1]}
 ]
}
```

A crystal structure has no pLDDT, so neither node nor contact carries a `confidence`. Thr670 is held by two
contacts: a line (weight 1.0) and a hydrophobic sightline (0.35).
