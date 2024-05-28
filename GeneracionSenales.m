i=1;
conapnea = 0;
sinapnea = 0;

fs = 100;
duracion = 10;
t = 0:1/fs:(duracion-1/fs);
TconApnea = table();
TconApnea.tiempo = t';
TsinApnea = table();
TsinApnea.tiempo = t';

while(i < 4001)

frecuencia1 = randi([1,4])/10;
amplitud1 = 0.1; 
senal1 = amplitud1 * sin(2 * pi * frecuencia1 * t);

frecuencia2 = rand();
amplitud2 = 0.05; 
senal2 = amplitud2 * cos(2 * pi * frecuencia2 * t); 

frecuencia3 = rand();
amplitud3 = 0.05; 
senal3 = amplitud3 * cos(2 * pi * frecuencia3 * t); 

senal = senal1 + senal2 + senal3;


%% PICOS ANCHOS

cant_picos = randi([1,3]);
while cant_picos > 0
    
    corrimiento = (duracion-1/fs)*rand();
    ancho = 10 + (20 - 1)*rand();
    amplitud = -0.5 + rand();
    
    deformacion = -1 + 2* rand();

    pico = amplitud * tripuls(t-corrimiento,ancho, deformacion);
    
    senal = senal + pico;
    
    cant_picos = cant_picos - 1;

end

%% RUIDO

cant_ruido = randi([1,3]);
while cant_ruido > 0
    
    nivel_ruido = 1/randi([10,15]); 

    ruido = nivel_ruido * randn(size(t));
    senal = senal + ruido;
    cant_ruido = cant_ruido - 1;
    
end


%% PICOS DE APNEA
apneas = false;

apnea_draw = rand;
porcentaje = 0.5;
apnea_flag = 0;

if apnea_draw > porcentaje
    apnea_flag = 1;
end
    

if apnea_flag == 1;
    cant_apneas = randi([1,3]);
else
    cant_apneas = 0;
end
    
while cant_apneas > 0
    corrimiento = (duracion-1/fs)*rand();
    ancho = 1 + (4 - 1)*rand();
    modulo_amplitud = 0.5+1.5*rand();
    signo_amplitud = round(rand());
    if(signo_amplitud == 1)
        amplitud = modulo_amplitud;
    else
        amplitud = -modulo_amplitud;
    end
    pico = amplitud * tripuls(t-corrimiento,ancho);
    
    senal = senal + pico;
    apneas = true;
    cant_apneas = cant_apneas - 1;
        
end

if(apneas)
    conapnea = conapnea + 1;
    TconApnea.(strcat('Senal', num2str(conapnea))) = senal';
else
    sinapnea = sinapnea + 1;
    TsinApnea.(strcat('Senal', num2str(sinapnea))) = senal';
end

i = i+1;

end
writetable(TconApnea, 'SenalesCONapnea.csv')
writetable(TsinApnea, 'SenalesSINapnea.csv')