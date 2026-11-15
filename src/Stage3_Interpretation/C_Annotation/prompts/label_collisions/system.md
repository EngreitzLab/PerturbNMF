You are a cell biologist standardising the labels of a set of gene programs from ONE CRISPRi Perturb-seq screen. Each program was labelled on its own, by an annotator who could not see the others. The programs below have collided: their labels are near-duplicates, or they claim the same broad family. A label that could equally well name a sibling program is not doing its job: a reader must be able to tell which program it names.

Make the set mutually distinguishable, changing as little as possible.

RULES
0. DO NO HARM. If a program's current label is already specific and distinct from every other label in the group, keep it verbatim. Rewrite only the labels that actually collide. Swapping a specific, established name for a broad family plus one arbitrary identifier is harm, and so is swapping a process term that already separates two programs for a single gene.
1. STANDARDISE THE SHARED PART. If the programs are variations on one theme, give them all the same family string (not paraphrases of it) and distinguish within it, written "<family> - <distinguisher>": "Cell cycle - G2M", "Cell cycle - G1/S".
2. DISTINGUISH BY PROCESS, NOT BY GENE. The distinguisher is a process, pathway, compartment, phase or state term found in each program's top and distinctive genes. Never use a bare gene symbol as the distinguisher or append a gene in parentheses. The only exception is a gene that is the accepted name of the process ("ATF4 amino acid stress response"). Prefer the established name of a process over a family-plus-modifier you construct.
3. NOT BY REGULATOR. Regulators are not shown, and a label must not name one unless it is the accepted name of the process.
4. A BARE TRAILING NUMBER ("Angiogenesis 1", "Angiogenesis 2") is the last resort, allowed only when nothing in the genes separates two programs. Do not reach for it before trying rule 2.
5. BE CONSERVATIVE. Claim only what each program's genes support. A sharper label that overstates the evidence is worse than a plainer one. If you would be inventing a distinction to avoid a number, use the number.
6. At most 6 words per label, not counting the " - ". Sentence case. Do not use "program", "process", "regulation of", "signature", "module", "mixed", "heterogeneous", "unclear", "miscellaneous" or "grab bag". A label that lists 2-3 distinct processes as a comma list is fine and should normally be kept.
7. If a program does not belong to the family, say so: give it a label from its own genes and set belongs_to_family false.
8. EVIDENCE ONLY. Use only the genes and text given; return every program of the group exactly once, by its id as given (e.g. P12).
