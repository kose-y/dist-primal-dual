
import os, os.path, sys
if sys.platform=='linux':
    urls = ["https://www.dropbox.com/s/d7tpa8insoq844g/ogrp_100_100_10_5000_X.mat",
        "https://www.dropbox.com/s/t14b7p4dq66up9k/ogrp_100_100_10_5000.mat",
        "https://www.dropbox.com/s/jc96pvzq4lo58ot/Zhu_1000_10_5000_20_0.7_100_X.mat",
        "https://www.dropbox.com/s/tl8ode7ny8elole/Zhu_1000_10_5000_20_0.7_100.mat"
    ]
    files = ["data/ogrp_100_100_10_5000_X.mat",
        "data/ogrp_100_100_10_5000.mat",
        "data/Zhu_1000_10_5000_20_0.7_100_X.mat",
        "data/Zhu_1000_10_5000_20_0.7_100.mat"
    ]

    for url, f in zip(urls, files):
        if not os.path.exists(f):
            os.system("wget --directory-prefix=data/ {}".format(url))
else:
    print("This script currently only supports linux.")
