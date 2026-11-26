"""What is written on the atoms that are drawn of a residue (``render(atom_labels=...)``), and that it is right."""

import math
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

import pytest

from conftest import TESTDATA, chain_diagram, diagram_of, model_of, nucleic_diagram, render_of
from bindview import load_interactions
from bindview.cli import build_parser
from bindview.fragments import build_fragment
from bindview.residues import (_NUCLEOTIDE_TEMPLATES, _TEMPLATES, canonical_atom, canonical_resname, element_of, template_bonds,
                               template_for, written)
from bindview.structure import read_structure

SVG = "{http://www.w3.org/2000/svg}"
ALL = (*_TEMPLATES, *_NUCLEOTIDE_TEMPLATES)
#: The radius of an element, in ångström: two atoms are bonded that are as near as the sum of theirs, and 0.4.
RADIUS = {"C": 0.77, "N": 0.73, "O": 0.71, "S": 1.04, "P": 1.10}


def words(group: ET.Element) -> "list[str]":
    return ["".join(t.itertext()) for t in group.iter(f"{SVG}text")]


def atoms_written(svg: str) -> "Counter[str]":
    """The words on the atoms of the residues: what stands in a group of its own for the atoms of one."""
    out: "Counter[str]" = Counter()
    for g in ET.fromstring(svg).iter(f"{SVG}g"):
        if g.get("class") == "fragment":
            out.update(words(g))
    return out


def test_every_name_and_element_and_bond_is_that_of_the_structures():
    """The twenty amino acids and the four ribonucleotides, against the residues of three entries of the PDB and of
    the riboswitch that are in the repository."""
    files = [model_of(case) for case in ("kit_imatinib", "dhfr_methotrexate", "thrombin_hirudin")]
    files += [TESTDATA / "nucleic" / "riboswitch_bisguanidine" / "model.pdb"]
    seen: "Counter[str]" = Counter()
    whole = 0
    for file in files:
        residues = defaultdict(dict)
        for a in read_structure(file).atoms:
            if canonical_resname(a.resname) in ALL and a.element not in ("H", "D"):
                residues[a.chain, a.resnum, a.icode, a.resname][canonical_atom(a.resname, a.name)] = a
        whole += len(residues)
        for (_, _, _, resname), mine in residues.items():
            template = template_for(resname)
            if not set(template.atoms) <= set(mine):
                continue                                         # the end of a chain, a side chain that is not modelled
            assert set(mine) - set(template.atoms) <= {"OXT", "OP3"}, (resname, set(mine) - set(template.atoms))
            for name in template.atoms:
                assert element_of(name) == mine[name].element, (resname, name)
            near = {frozenset((p, q)) for p in template.atoms for q in template.atoms if p < q and
                    math.dist((mine[p].x, mine[p].y, mine[p].z), (mine[q].x, mine[q].y, mine[q].z))
                    < RADIUS[mine[p].element] + RADIUS[mine[q].element] + 0.40}
            assert near == {frozenset(b[:2]) for b in template_bonds(resname)}, resname
            seen[canonical_resname(resname)] += 1
    # Every kind is seen, and all but a few residues of the files are held to their templates.
    assert set(_TEMPLATES) | {"A", "C", "G", "U"} <= set(seen) and sum(seen.values()) > 0.9 * whole


def test_what_is_written_on_each_atom():
    for resname in ALL:
        frag = build_fragment(resname, template_for(resname).atoms, 1.0, detail="atoms")
        for name in frag.atoms:
            assert written(name) == written(name, "names") == name
            assert written(name, "elements") == frag.elements[name] and len(frag.elements[name]) == 1
    lys = build_fragment("LYS", ("NZ",), 1.0)
    assert [written(a, "elements") for a in lys.atoms if lys.elements[a] != "C"] == ["N"]
    # Names that can be read as something else than what they are: zinc, neon, an NH and an NH2, a hydroxyl.
    assert {n: written(n, "elements") for n in ("NZ", "NE", "NH1", "NH2", "OH", "OG", "SD", "O4'", "O1P", "N7")} == \
        {"NZ": "N", "NE": "N", "NH1": "N", "NH2": "N", "OH": "O", "OG": "O", "SD": "S", "O4'": "O", "O1P": "O", "N7": "N"}


def places(result) -> "list[tuple]":
    return [(n.id, n.x, n.y, n.attrs.get("angle")) for n in result.graph.nodes]


@pytest.mark.parametrize("case", ["kit_imatinib", "dhfr_methotrexate"])
def test_the_letter_of_the_element_takes_the_place_of_the_name_and_nothing_moves(case):
    plain, res = render_of(case, atom_labels="names"), render_of(case)
    names, letters = atoms_written(plain.svg), atoms_written(res.svg)
    assert any(len(n) > 1 for n in names) and set(letters) <= {"N", "O", "S"}
    assert Counter(element_of(n) for n in names.elements()) == letters            # one for one
    assert places(res) == places(plain) and res.quality == plain.quality and res.atoms == plain.atoms
    assert diagram_of(case).render(atom_labels="elements").svg == res.svg                  # (the default)
    # Whoever wants the name finds it where it was: in the tooltip of the contact, and in its data.
    def said(svg: str) -> "list[tuple]":
        return [(g.get("class"), g.get("data-protein-atoms"), g.findtext(f"{SVG}title")) for g in ET.fromstring(svg).iter(f"{SVG}g")
                if g.get("class", "").split(" ")[0] in ("interaction", "contact")]
    assert said(res.svg) == said(plain.svg) and any(a and len(a) > 1 for _, a, _ in said(res.svg))


def test_a_line_ends_on_the_letter_and_not_where_the_name_would_end():
    plain, res = render_of("dhfr_methotrexate", atom_labels="names"), render_of("dhfr_methotrexate")

    def length(svg: str) -> float:
        g = next(g for g in ET.fromstring(svg).iter(f"{SVG}g") if g.get("class") == "interaction hbond"
                 and len(g.get("data-protein-atoms")) == 3)
        at = [float(line.get(k)) for line in g.iter(f"{SVG}line") for k in ("x1", "y1", "x2", "y2")]
        return max(math.dist(p, q) for p in zip(at[::2], at[1::2]) for q in zip(at[::2], at[1::2]))
    assert length(res.svg) > length(plain.svg)


def test_it_is_so_for_nucleotides_for_a_chain_and_for_what_is_shown_while_it_is_pointed_at():
    rna, plain = nucleic_diagram().render(), nucleic_diagram().render(atom_labels="names")
    assert set(atoms_written(rna.svg)) <= {"N", "O", "P"} and places(rna) == places(plain)
    assert {"N7", "O6", "O1P", "O4'"} <= set(atoms_written(plain.svg))
    # The residues beside a peptide (hirudin on thrombin), and the atoms that stand beside the boxes of its own residues.
    peptide = chain_diagram("thrombin_hirudin", detail="sidechain")
    chain, before = (peptide.render(atom_labels=a, on_unreadable="render") for a in ("elements", "names"))
    assert places(chain) == places(before) and chain.quality == before.quality
    assert "NZ" not in words(ET.fromstring(chain.svg)) and set(atoms_written(chain.svg)) <= {"N", "O", "S"}
    beside = [w for g in ET.fromstring(chain.svg).iter(f"{SVG}g") if g.get("class") == "atoms" for w in words(g)]
    assert beside and all(re.fullmatch(r"[NOS]|[A-Z]?[A-Za-z0-9₀-₉⁺⁻−+]*", w) for w in beside)
    assert not {"OD1", "OD2", "OE1", "OE2", "NE", "NH1", "NH2", "OH", "OG"} & set(words(ET.fromstring(chain.svg)))
    # What a bubble shows while it is pointed at: none of the names of imatinib's contact atoms in its table, and the
    # letters of them (Cys673 holds a sulphur).
    live = diagram_of("kit_imatinib").render(style="schematic", interactive=True).svg
    more = [w for g in ET.fromstring(live).iter(f"{SVG}g") if g.get("class", "").startswith("more") for w in words(g)]
    named = {r.protein_atom for r in load_interactions(TESTDATA / "kit_imatinib" / "interactions.tsv")
             if r.protein_atom and len(r.protein_atom) > 1}
    assert more and named and not named & set(more) and {"N", "O", "S"} <= set(more)


def test_it_is_asked_for_by_its_words():
    with pytest.raises(ValueError, match="atom_labels is one of 'elements', 'names', not 'letters'"):
        diagram_of("kit_imatinib").render(atom_labels="letters")
    assert build_parser().parse_args(["render", "x.cif", "--ligand", "L"]).atom_labels == "elements"
    assert build_parser().parse_args(["render", "x.cif", "--ligand", "L", "--atom-labels", "names"]).atom_labels == "names"
