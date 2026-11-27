# Changelog

## 0.1.1 - 2026-11-27

- `bindview compare --min-fraction`: how much of the smallest ligand the common substructure covers to be aligned on, as `Comparison(min_fraction=)` takes it (0.6).
- A colour scheme of your own, light and dark, registered in `bindview.theme.SCHEMES` and asked for by its name (`colors=`), with every theme, `"auto"` too: in the manual, and tested.
- Fixed: a contact named by two or more backbone atoms of an arginine, aspartate, glutamate, lysine or histidine (`N` and `O`, say) raised an `IndexError` in the detailed style; its line now stops at those atoms.

## 0.1.0 - 2026-11-26

bindview draws molecular interaction diagrams as SVG, from a structure file and a table of contacts ([manual](docs/manual.md)).

- A ligand in its pocket, in a detailed style (the atoms that make each contact) and a schematic one (residues as bubbles), or residue by residue in either. Every figure is checked for overlaps and for legible text on the page it is for; one that cannot be read is replaced by a table of its contacts that says why.
- Contacts from a table (tab, comma or semicolon between the fields), from PLIP where it is installed (optional, GPL-2.0, `bindview[plip]`), or from a plain table of heavy-atom distances that needs nothing else (`interactions="distances"`).
- Peptides and strands of RNA or DNA as ligands drawn as chains, ligands in RNA or DNA, covalent ligands, ions that bridge ligand and protein, and predicted water sites, marked as predicted.
- The interface of two chains (`Interface`), with the predicted aligned error of each contact (AlphaFold 3, Boltz, the AlphaFold database with its complexes) and ipSAE; a chain found by the UniProt accession its mmCIF file assigns it (`Structure.chain_of`); two strands of RNA or DNA and their base pairs (`BasePairs`).
- The confidence of a predicted model where the file declares it: pLDDT by line weight or in AlphaFold's colours, ipTM, PAE. The ligand's buried fraction.
- Several ligands of one pocket (`Comparison`: what differs, what they share, fingerprints as CSV) and several models of one complex (`Ensemble`: the share of the models that make each contact, all of them in one figure).
- The layout of a figure as a graph, written as JSON; that of a ligand diagram or of base pairs can be changed and drawn again (`render(graph=...)`, or by dragging in a browser with `Diagram.arrange()`). Figures that answer the pointer (`interactive=True`), light, dark or following the reader (`theme="auto"`), in two colour schemes (`colors="contrast"`).
- The command `bindview`: `render`, `interface`, `basepairs`, `compare`, `fingerprint` and `report`.
