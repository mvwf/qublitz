from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

# Standard refers to the ones we are studying in the paper.
class DegeneracyType(Enum):
    # The ones studied in the paper
    PRIMARY_EP = "PRIMARY_EP"
    PRIMARY_TPD = "PRIMARY_TPD"
    # The alternative pair, the dual of the ones in the paper
    SECONDARY_EP = "SECONDARY_EP"
    SECONDARY_TPD = "SECONDARY_TPD"
    # TPDs not on the same path as an EP
    ROGUE_TPD = "ROGUE_TPD"

@dataclass
class Degeneracy:
    degeneracy_type: DegeneracyType
    # Location
    Delta_tilde_f: Optional[float]
    Delta_tilde_kappa: Optional[float]

def ep_location(phi) -> list[Degeneracy]:
    # Two EPs at each phi
    if phi < np.pi:
        return [
            Degeneracy(
                degeneracy_type=DegeneracyType.PRIMARY_EP,
                Delta_tilde_kappa=-2 * np.cos(phi / 2),
                Delta_tilde_f=-2 * np.sin(phi / 2)
            ),
            Degeneracy(
                degeneracy_type=DegeneracyType.SECONDARY_EP,
                Delta_tilde_kappa=2 * np.cos(phi / 2),
                Delta_tilde_f=2 * np.sin(phi / 2)
            )
        ]
    else:
        return [
            Degeneracy(
                degeneracy_type=DegeneracyType.PRIMARY_EP,
                Delta_tilde_kappa= 2 * np.cos(phi / 2),
                Delta_tilde_f= 2 * np.sin(phi / 2)
            ),
            Degeneracy(
                degeneracy_type=DegeneracyType.SECONDARY_EP,
                Delta_tilde_kappa= -2 * np.cos(phi / 2),
                Delta_tilde_f= -2 * np.sin(phi / 2)
            )
        ]

def tpd_location(phi, kappa_tilde_c) -> list[Degeneracy]:
    # There can be up to 6 TPDs at each phi. Usually there are less.
    # Check if phi / pi is in Z
    degen_list = []
    if phi == 0:
        # There can either be 0, 2, or 6 TPDs
        degen_list = __tpd_location_phi_zero(kappa_tilde_c)
    elif phi == np.pi:
        # Always 4 TPDs
        degen_list = __tpd_location_phi_pi(kappa_tilde_c)
    else:
        # Any other phi
        degen_list = __tpd_location_any_phase(phi, kappa_tilde_c)

    return degen_list


def __tpd_location_phi_pi(kappa_tilde_c) -> list[Degeneracy]:
    degen_list = [
        Degeneracy(
            degeneracy_type=DegeneracyType.PRIMARY_TPD,
            Delta_tilde_kappa=0,
            Delta_tilde_f=np.sqrt(kappa_tilde_c**2 + 4)
        ),
        Degeneracy(
            degeneracy_type=DegeneracyType.SECONDARY_TPD,
            Delta_tilde_kappa=0,
            Delta_tilde_f=-np.sqrt(kappa_tilde_c**2 + 4)
        ),
        Degeneracy(
            degeneracy_type=DegeneracyType.ROGUE_TPD,
            Delta_tilde_kappa=kappa_tilde_c,
            Delta_tilde_f=np.sqrt(kappa_tilde_c**2 + 4)
        ),
        Degeneracy(
            degeneracy_type=DegeneracyType.ROGUE_TPD,
            Delta_tilde_kappa=kappa_tilde_c,
            Delta_tilde_f=-np.sqrt(kappa_tilde_c**2 + 4)
        )
    ]

    return degen_list

def __tpd_location_any_phase(phi, kappa_tilde_c):
    # First, add the 2 Rogue TPDs
    degen_list = []
    if kappa_tilde_c**2 >= 4 * np.cos(phi):
        degen_list.extend([
            Degeneracy(
                degeneracy_type=DegeneracyType.ROGUE_TPD,
                Delta_tilde_kappa=kappa_tilde_c,
                Delta_tilde_f=np.sqrt(kappa_tilde_c**2 - 4 * np.cos(phi))
            ),
            Degeneracy(
                degeneracy_type=DegeneracyType.ROGUE_TPD,
                Delta_tilde_kappa=kappa_tilde_c,
                Delta_tilde_f=-np.sqrt(kappa_tilde_c**2 - 4 * np.cos(phi))
            )
        ])

    # Now, add the 2 TPDs (maybe there can be 4?)
    possible_dks = np.roots([2, -2 * kappa_tilde_c, kappa_tilde_c **2 - 4 * np.cos(phi), 0, 4 * np.cos(phi)**2 - 4])
    # Throw out the complex roots - only get the real root with negative real part
    real_roots = possible_dks[np.abs(possible_dks.imag) < 1e-10].real
    for x in real_roots:
        corresponding_df = -1 * (-kappa_tilde_c**2 * x + 2 * kappa_tilde_c * x**2 - 2 * x**3 + 4 * np.cos(phi) * x) / (2 * np.sin(phi))
        degen_list.append(
            Degeneracy(
                degeneracy_type=DegeneracyType.PRIMARY_TPD if x < 0 else DegeneracyType.SECONDARY_TPD,
                Delta_tilde_kappa=x,
                Delta_tilde_f=corresponding_df
            )
        )

    return degen_list

def __tpd_location_phi_zero(kappa_tilde_c) -> list[Degeneracy]:
    degen_list = []
    if kappa_tilde_c ** 2 <= 8:
        degen_list.extend([
            Degeneracy(
                degeneracy_type=DegeneracyType.PRIMARY_TPD,
                Delta_tilde_kappa=( kappa_tilde_c - np.sqrt(8 - kappa_tilde_c ** 2) ) / 2,
                Delta_tilde_f=0
            ),
            Degeneracy(
                degeneracy_type=DegeneracyType.SECONDARY_TPD,
                Delta_tilde_kappa=( kappa_tilde_c + np.sqrt(8 - kappa_tilde_c ** 2) ) / 2,
                Delta_tilde_f=0
            )
        ])
    if kappa_tilde_c >= 2:
        # 4 more TPDs open up here! For a total of 6!
        degen_list.extend([
            Degeneracy(
                degeneracy_type=DegeneracyType.ROGUE_TPD,
                Delta_tilde_kappa=0,
                Delta_tilde_f=np.sqrt(kappa_tilde_c**2 - 4)
            ),
            Degeneracy(
                degeneracy_type=DegeneracyType.ROGUE_TPD,
                Delta_tilde_kappa=0,
                Delta_tilde_f=-np.sqrt(kappa_tilde_c**2 - 4)
            ),
            Degeneracy(
                degeneracy_type=DegeneracyType.ROGUE_TPD,
                Delta_tilde_kappa=kappa_tilde_c,
                Delta_tilde_f=np.sqrt(kappa_tilde_c**2 - 4),
            ),
            Degeneracy(
                degeneracy_type=DegeneracyType.ROGUE_TPD,
                Delta_tilde_kappa=kappa_tilde_c,
                Delta_tilde_f=-np.sqrt(kappa_tilde_c**2 - 4),
            )
        ])

    return degen_list

