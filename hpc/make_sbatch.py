"""Generate the training sbatch files from one template.

Edit this, not the .sbatch files. Runs that are compared against each other
must differ in exactly one variable; generating them from a single template is
how that stays true. An earlier set of hand-written scripts silently drifted --
two runs used 2xs16+4xs8 while the others used 1xs16+8xs8, and one used batch 12
instead of 16, so three variables moved at once and no comparison between them
was valid.
"""
import os

TEMPLATE = open(os.path.join(os.path.dirname(__file__), '_template.sbatch')).read()

RUNS = [
    ('v3_FlyingChairs', 'nf3-chairs', 'FlyingChairs', '',
     'training data = FlyingChairs only (no driving imagery: the clean v2 comparison)'),
    ('v3_FlyingChairs_VKITTI2', 'nf3-ch-vk', 'FlyingChairs+VKITTI2', '',
     'training data = FlyingChairs + VKITTI2 (Scene01/02/06 only)'),
    ('v3_FlyingChairs_VKITTI2_Sintel', 'nf3-ch-vk-si', 'FlyingChairs+VKITTI2+Sintel', '',
     'training data = FlyingChairs + VKITTI2 + MPI-Sintel'),
    ('v3_FlyingChairs_VKITTI2_Sintel_Spring', 'nf3-ch-vk-si-sp',
     'FlyingChairs+VKITTI2+Sintel+Spring', '',
     'training data = FlyingChairs + VKITTI2 + MPI-Sintel + Spring'),
    ('v3_FlyingChairs_VKITTI2_Sintel_uncertainty', 'nf3-unc',
     'FlyingChairs+VKITTI2+Sintel', ' --uncertainty',
     'uncertainty head ON (same data as v3_FlyingChairs_VKITTI2_Sintel)'),
]

if __name__ == '__main__':
    d = os.path.dirname(__file__)
    for name, job, stage, extra, vary in RUNS:
        fn = os.path.join(d, f'{name}.sbatch')
        open(fn, 'w').write(TEMPLATE.format(name=name, job=job, stage=stage,
                                            extra=extra, vary=vary))
        os.chmod(fn, 0o755)
        print('wrote', fn)
