# native(1N8Z) vs s108 CDR overlay
load /home/kyeongtak/structure_projects/esm_binder_design_base/runs/rank_ce_out/1N8Z.cif, native
load /home/kyeongtak/structure_projects/esm_binder_design_base/runs/rank_ce_out/s108_exp_full_2021.cif, s108
hide everything
# 항체 Fv 만 (native chain A/B, s108 chain S)
create nat_fv, native and chain A+B
create s108_fv, s108 and chain S
delete native
delete s108
align s108_fv, nat_fv
show cartoon, nat_fv or s108_fv
color grey70, nat_fv
color slate, s108_fv
set cartoon_transparency, 0.3, nat_fv
bg_color white
