BEGIN;

DO $$
BEGIN
    IF current_setting('list_engine.app_mode', true) IS DISTINCT FROM 'demo' THEN
        RAISE EXCEPTION 'Demo seed refused: set list_engine.app_mode=demo explicitly';
    END IF;
END;
$$;

SET LOCAL search_path = list_engine, public;

INSERT INTO companies (
    piva, legal_name, website, ateco_code, city, province, region,
    revenue_eur, employees, company_status, source, is_demo, source_observed_at
)
VALUES
    ('99000000002', 'Aurora Logistica Demo S.r.l.', 'https://aurora-logistica.example.invalid', '49.41', 'Milano', 'MI', 'Lombardia', 6500000, 28, 'active', 'demo_seed', true, '2026-07-01T00:00:00Z'),
    ('99000000010', 'Boreale Trasporti Demo S.r.l.', 'https://boreale-trasporti.example.invalid', '49.41', 'Bergamo', 'BG', 'Lombardia', 12100000, 54, 'active', 'demo_seed', true, '2026-07-01T00:00:00Z'),
    ('99000000028', 'Cobalto Edilizia Demo S.p.A.', 'https://cobalto-edilizia.example.invalid', '41.20', 'Brescia', 'BS', 'Lombardia', 19800000, 83, 'active', 'demo_seed', true, '2026-07-01T00:00:00Z'),
    ('99000000036', 'Dalia Impianti Demo S.r.l.', 'https://dalia-impianti.example.invalid', '43.21', 'Verona', 'VR', 'Veneto', 4200000, 19, 'active', 'demo_seed', true, '2026-07-01T00:00:00Z'),
    ('99000000044', 'Edera Retail Demo S.r.l.', 'https://edera-retail.example.invalid', '47.19', 'Padova', 'PD', 'Veneto', 8700000, 42, 'active', 'demo_seed', true, '2026-07-01T00:00:00Z'),
    ('99000000051', 'Futura Packaging Demo S.r.l.', 'https://futura-packaging.example.invalid', '22.22', 'Vicenza', 'VI', 'Veneto', 14300000, 61, 'active', 'demo_seed', true, '2026-07-01T00:00:00Z'),
    ('99000000069', 'Ginestra Servizi Demo S.r.l.', 'https://ginestra-servizi.example.invalid', '81.22', 'Parma', 'PR', 'Emilia-Romagna', 3100000, 35, 'active', 'demo_seed', true, '2026-07-01T00:00:00Z'),
    ('99000000077', 'Helios Meccanica Demo S.r.l.', 'https://helios-meccanica.example.invalid', '25.62', 'Modena', 'MO', 'Emilia-Romagna', 23400000, 96, 'active', 'demo_seed', true, '2026-07-01T00:00:00Z'),
    ('99000000085', 'Iris Alimentare Demo S.r.l.', 'https://iris-alimentare.example.invalid', '10.89', 'Bologna', 'BO', 'Emilia-Romagna', 7600000, 47, 'active', 'demo_seed', true, '2026-07-01T00:00:00Z'),
    ('99000000093', 'Lario Hospitality Demo S.r.l.', 'https://lario-hospitality.example.invalid', '55.10', 'Como', 'CO', 'Lombardia', 5200000, 31, 'active', 'demo_seed', true, '2026-07-01T00:00:00Z'),
    ('99000000101', 'Marea E-commerce Demo S.r.l.', 'https://marea-commerce.example.invalid', '47.91', 'Treviso', 'TV', 'Veneto', 9700000, 38, 'active', 'demo_seed', true, '2026-07-01T00:00:00Z'),
    ('99000000119', 'Nebula Arredi Demo S.r.l.', 'https://nebula-arredi.example.invalid', '31.09', 'Monza', 'MB', 'Lombardia', 11700000, 51, 'active', 'demo_seed', true, '2026-07-01T00:00:00Z'),
    ('99000000127', 'Olmo Costruzioni Demo S.r.l.', 'https://olmo-costruzioni.example.invalid', '41.20', 'Torino', 'TO', 'Piemonte', 16500000, 72, 'active', 'demo_seed', true, '2026-07-01T00:00:00Z'),
    ('99000000135', 'Prisma Autotrasporti Demo S.r.l.', 'https://prisma-autotrasporti.example.invalid', '49.41', 'Novara', 'NO', 'Piemonte', 5800000, 26, 'active', 'demo_seed', true, '2026-07-01T00:00:00Z'),
    ('99000000143', 'Quercia Ferramenta Demo S.r.l.', 'https://quercia-ferramenta.example.invalid', '47.52', 'Udine', 'UD', 'Friuli-Venezia Giulia', 2400000, 14, 'active', 'demo_seed', true, '2026-07-01T00:00:00Z'),
    ('99000000150', 'Radura Tessile Demo S.r.l.', 'https://radura-tessile.example.invalid', '13.20', 'Prato', 'PO', 'Toscana', 13200000, 59, 'active', 'demo_seed', true, '2026-07-01T00:00:00Z'),
    ('99000000168', 'Selva Distribuzione Demo S.r.l.', 'https://selva-distribuzione.example.invalid', '46.90', 'Genova', 'GE', 'Liguria', 22100000, 88, 'active', 'demo_seed', true, '2026-07-01T00:00:00Z'),
    ('99000000176', 'Tundra Facility Demo S.r.l.', 'https://tundra-facility.example.invalid', '81.10', 'Varese', 'VA', 'Lombardia', 6800000, 64, 'active', 'demo_seed', true, '2026-07-01T00:00:00Z'),
    ('99000000184', 'Ulivo Manifattura Demo S.r.l.', 'https://ulivo-manifattura.example.invalid', '32.99', 'Mantova', 'MN', 'Lombardia', 10100000, 44, 'active', 'demo_seed', true, '2026-07-01T00:00:00Z'),
    ('99000000192', 'Vela Ricambi Demo S.r.l.', 'https://vela-ricambi.example.invalid', '45.32', 'Rovigo', 'RO', 'Veneto', 3900000, 21, 'active', 'demo_seed', true, '2026-07-01T00:00:00Z')
ON CONFLICT (piva) DO NOTHING;

INSERT INTO company_firmographic_history (
    piva, revenue_eur, employees, company_status, source, valid_from
)
SELECT piva, revenue_eur, employees, company_status, source, source_observed_at
FROM companies
WHERE source = 'demo_seed'
  AND piva = ANY (ARRAY[
      '99000000002', '99000000010', '99000000028', '99000000036',
      '99000000044', '99000000051', '99000000069', '99000000077',
      '99000000085', '99000000093', '99000000101', '99000000119',
      '99000000127', '99000000135', '99000000143', '99000000150',
      '99000000168', '99000000176', '99000000184', '99000000192'
  ])
ON CONFLICT (piva, source, valid_from) DO NOTHING;

COMMIT;
