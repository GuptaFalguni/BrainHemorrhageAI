BHAI demo CT scans
==================

Research CTs from BHSD (Wu et al., arXiv:2308.11298).
Not for clinical use. Do not share as a clinical dataset.

How to use
----------
After Docker is running, open http://localhost:3000 and upload a .nii.gz
from a numbered folder. Start with MONAI.

  01_all_classes\all_classes.nii.gz
      Only scan here with all five subtypes (EDH + SDH + SAH + IPH + IVH).
      Use this first in a presentation.

  02_four_classes\
      Large, easy-to-see scans missing exactly one subtype.

  03_by_class\
      Two examples per subtype. _1 is the larger / clearer case.

Do not upload
-------------
  *_WHAT_IS_HERE.txt
  MANIFEST.csv
  this README
  ground_truth\   (expert masks for notes only)

Class key
---------
1 EDH  epidural
2 SDH  subdural
3 SAH  subarachnoid
4 IPH  intraparenchymal
5 IVH  intraventricular
