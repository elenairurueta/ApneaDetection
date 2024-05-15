i=1;
conapnea = 0;
sinapnea = 0;

fs = 200;
duracion = 30;
t = 0:1/fs:duracion;
TconApnea = table();
TconApnea.tiempo = t';
TsinApnea = table();
TsinApnea.tiempo = t';

while(i < 2000)

%frecuencia1 = 6; 
frecuencia1 = randi([1,4])/10;
amplitud1 = 0.1; 
senal1 = amplitud1 * sin(2 * pi * frecuencia1 * t);

%frecuencia2 = 2; 
frecuencia2 = rand();
amplitud2 = 0.05; 
senal2 = amplitud2 * cos(2 * pi * frecuencia2 * t); 

%frecuencia3 = 2; 
frecuencia3 = rand();
amplitud3 = 0.05; 
senal3 = amplitud3 * cos(2 * pi * frecuencia3 * t); 

% %frecuencia4 = 2; 
% frecuencia4 = randi([10,15]);
% amplitud4 = 0.2; 
% senal4 = amplitud4 * sin(2 * pi * frecuencia4 * t); 

senal = senal1 + senal2 + senal3;


%% PICOS ANCHOS

cant_picos = randi([1,3]);
while cant_picos > 0
    
    %corrimiento = 500;
    corrimiento = duracion*rand();
    %ancho = 100;
    ancho = 10 + (20 - 1)*rand();
    %amplitud = -1;
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
cant_apneas = randi([0,3]);

while cant_apneas > 0
    
    %corrimiento = 500;
    corrimiento = duracion*rand();
    %ancho = 100;
    ancho = 1 + (4 - 1)*rand();
    %amplitud = -1;
    amplitud = -2 + 4*rand();
    if(abs(amplitud) > 0.2)
        apneas = true;
    end
    pico = amplitud * tripuls(t-corrimiento,ancho);
    
    senal = senal + pico;
    
    cant_apneas = cant_apneas - 1;
        
end

%% gráfico
% figure(1)
% subplot(7,4,i)
% plot(t, senal, 'r');
% xlim([0, duracion]);
% ylim([-3, 3]);
% set(gca,'XTick',[],'YTick',[]);

if(apneas)
    %xlabel('CON apnea');
    conapnea = conapnea + 1;
    TconApnea.(strcat('Senal', num2str(conapnea))) = senal';
else
    %xlabel('SIN apnea');
    sinapnea = sinapnea + 1;
    TsinApnea.(strcat('Senal', num2str(sinapnea))) = senal';
end

i = i+1;

end
writetable(TconApnea, 'SenalesCONapnea.csv')
writetable(TsinApnea, 'SenalesSINapnea.csv')