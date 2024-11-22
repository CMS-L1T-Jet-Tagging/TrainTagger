import matplotlib.pyplot as plt

def plot_ratio(all_events, selected_events, plot=False):
    fig = plt.figure(figsize=(10, 12))
    _, eff = selected_events.plot_ratio(all_events,
                                              rp_num_label="Selected events", rp_denom_label=r"All Taus",
                                              rp_uncert_draw_type="bar", rp_uncertainty_type="efficiency")
    
    if plot: plt.savefig('plots/HH_eff_test.pdf', bbox_inches='tight')

    plt.show(block=False)
    return eff

def get_bar_patch_data(artists):
    x_data = [artists.bar.patches[i].get_x() for i in range(len(artists.bar.patches))]
    y_data = [artists.bar.patches[i].get_y() for i in range(len(artists.bar.patches))]
    err_data = [artists.bar.patches[i].get_height() for i in range(len(artists.bar.patches))]
    return x_data, y_data, err_data
