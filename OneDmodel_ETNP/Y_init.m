function Y=Y_init()

Y = zeros(1,24);

Y(1)      = 0.987846977687179 ; %0.987825793384144  ; %PK ; %'z_1 (dimensionless) (in CaRU_reduced_states)',
Y(2)      = 0.00889630179839756 ; %0.0088997523924806 ; %PK       ; %'z_2 (dimensionless) (in CaRU_reduced_states)',
Y(3)      = 0.00322801867751232 ; %0.00324558615058641; %PK       ; %'z_3 (dimensionless) (in CaRU_reduced_states)',
Y(4)       = 0.002510729584521 ; %0.0025137256732967 ; %PK   ; %'r1 (dimensionless) (r in Ca_independent_transient_outward_K_current_r_gate)',
Y(5)        = 0.992513467576987 ; %0.878224632545503  ; %PK ; %'s (dimensionless) (in Ca_independent_transient_outward_K_current_s_gate)',
Y(6)   = 0.695813093535168 ; %0.437771099997967  ; %PK ; %'s_slow (dimensionless) (in Ca_independent_transient_outward_K_current_s_slow_gate)',
Y(7)        = 0.00284312435079205 ; %0.00269405442932685; %PK ; %'y (dimensionless) (in hyperpolarisation_activated_current_y_gate)',
Y(8)    = 0.742474679225752 ; %0.746742321760617  ; %PK ; %'Ca_SR (mM) (in intracellular_ion_concentrations)',
Y(9)     = 0.000116263904902851 ; %0.000116580635095556 ; %PK ; %'Ca_i (mM) (in intracellular_ion_concentrations)',
Y(10)      = 137.631285943762 ; %137.893047365868   ; %PK      ; %'K_i (mM) (in intracellular_ion_concentrations)',
Y(11)     = 12.2406809965944 ; %12.0884900027087   ; %PK  ; %'Na_i (mM) (in intracellular_ion_concentrations)',
Y(12)     = 0.00104123225158751 ; %0.00104514167396484; %PK   ; %'TRPN (mM) (in intracellular_ion_concentrations)',
Y(13)        = -78.9449088406309 ; %-78.9312583049895  ; %PK   ; %'V (mV) (in membrane)',
Y(14)      = 0.0                       ; %'Q_1 (dimensionless) (in niederer_Cross_Bridges)',
Y(15)      = 0.0                      ; %'Q_2 (dimensionless) (in niederer_Cross_Bridges)',
Y(16)      = 0.0                       ; %'Q_3 (dimensionless) (in niederer_Cross_Bridges)',
Y(17)        = 0.00982273628901418 ; %PK       ; %'z (dimensionless) (in niederer_tropomyosin)', нет в этой модели
Y(18)        = 0.615054357972781 ; %0.614506673298225 ; %PK       ; %'h (dimensionless) (in sodium_current_h_gate)',
Y(19)        = 0.614459274494406 ; %0.613581967303456 ; %PK       ; %'j (dimensionless) (in sodium_current_j_gate)',
Y(20)    = 0.00536599273054972 ; %0.00537721194070032 ; %PK       ; %'m (dimensionless) (in sodium_current_m_gate)',
Y(21)     = 0.00331493233624982 ; %0.00331875525114995 ; %PK; %'r_ss (dimensionless) (in steady_state_outward_K_current_r_ss_gate)',
Y(22)     = 0.271740759186668 ; %0.270771693934212   ; %PK     ; %'s_ss (dimensionless) (in steady_state_outward_K_current_s_ss_gate)',
Y(23)       = 0.0; %'B1', быстрый обобщенный кальциевый буфер
Y(24)       = 0.0;           
