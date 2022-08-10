def get_cmap_vals(cmap_id, num):
    print('Colormap values for {}, {} pts'.format(cmap_id, num))
    cmap = plt.get_cmap(cmap_id, num)
    for i in range(cmap.N):
        rgba = cmap(i)
        print(i, rgba[:-1]) # Chopping off a value in m