import os

def criar_imagem_exemplo():
    largura, altura = 50, 50
    pixels = []
    
    for y in range(altura):
        linha = []
        for x in range(largura):
            r = int(255 * (x / largura))
            g = int(255 * (y / altura))
            b = int(255 * ((x + y) / (largura + altura)))
            linha.append((r, g, b))
        pixels.append(linha)
    
    return largura, altura, pixels

def converter_rgb_para_cinza_manual(r, g, b):
    return int(0.299 * r + 0.587 * g + 0.114 * b)

def aplicar_threshold_binario_manual(valor_cinza, threshold=128):
    return 255 if valor_cinza >= threshold else 0

def processar_imagem_completa(pixels_rgb, threshold=128):
    altura = len(pixels_rgb)
    largura = len(pixels_rgb[0]) if altura > 0 else 0
    
    pixels_cinza = []
    pixels_binarios = []
    
    for y in range(altura):
        linha_cinza = []
        linha_binaria = []
        
        for x in range(largura):
            r, g, b = pixels_rgb[y][x]
            
            cinza = converter_rgb_para_cinza_manual(r, g, b)
            linha_cinza.append(cinza)
            
            binario = aplicar_threshold_binario_manual(cinza, threshold)
            linha_binaria.append(binario)
        
        pixels_cinza.append(linha_cinza)
        pixels_binarios.append(linha_binaria)
    
    return pixels_cinza, pixels_binarios

def salvar_como_texto(pixels, nome_arquivo, tipo="cinza"):
    with open(nome_arquivo, 'w', encoding='utf-8') as arquivo:
        for linha in pixels:
            if tipo == "RGB":
                linha_str = ' '.join(f"({r},{g},{b})" for r, g, b in linha)
            else:
                linha_str = ' '.join(f"{pixel:3d}" for pixel in linha)
            arquivo.write(linha_str + '\n')

def visualizar_ascii(pixels, tipo="binario"):
    for linha in pixels:
        linha_ascii = ""
        for pixel in linha:
            if tipo == "binario":
                char = "█" if pixel == 0 else " "
            else:
                if pixel < 32:
                    char = "█"
                elif pixel < 64:
                    char = "▓"
                elif pixel < 96:
                    char = "▒"
                elif pixel < 128:
                    char = "░"
                elif pixel < 160:
                    char = "."
                elif pixel < 192:
                    char = ":"
                elif pixel < 224:
                    char = "-"
                else:
                    char = " "
            linha_ascii += char
        print(linha_ascii)

def analisar_estatisticas(pixels_originais, pixels_cinza, pixels_binarios):
    altura = len(pixels_originais)
    largura = len(pixels_originais[0])
    total_pixels = altura * largura
    
    pixels_pretos = sum(linha.count(0) for linha in pixels_binarios)
    pixels_brancos = total_pixels - pixels_pretos
    
    return pixels_pretos, pixels_brancos, total_pixels



def main():
    largura, altura, pixels_rgb = criar_imagem_exemplo()
    pixels_cinza, pixels_binarios = processar_imagem_completa(pixels_rgb, threshold=128)
    
    salvar_como_texto(pixels_rgb, "imagem_original.txt", "RGB")
    salvar_como_texto(pixels_cinza, "imagem_cinza.txt", "cinza")
    salvar_como_texto(pixels_binarios, "imagem_binaria.txt", "binario")
    
    pixels_pretos, pixels_brancos, total = analisar_estatisticas(pixels_rgb, pixels_cinza, pixels_binarios)
    
    return pixels_cinza, pixels_binarios

if __name__ == "__main__":
    main()