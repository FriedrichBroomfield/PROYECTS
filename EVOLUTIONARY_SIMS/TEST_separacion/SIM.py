from CLASES import *

# ============================================================
# Plot / UI (black background, no grid, white outer ring)
# ============================================================
sim = Simulation()
env = sim.env
pop = sim.pop

fig = plt.figure(figsize=(10.2, 7.2), facecolor="black")

ax_stats = fig.add_axes([0.02, 0.10, 0.30, 0.84], facecolor="black")
ax_stats.set_axis_off()
stats_txt = ax_stats.text(
    0.02, 0.98, "", va="top", ha="left",
    color="white", fontsize=10, family="monospace"
)

ax_slider = fig.add_axes([0.05, 0.04, 0.25, 0.03], facecolor="black")
speed_slider = Slider(ax_slider, "steps/frame", valmin=1, valmax=140, valinit=P["STEPS_PER_FRAME_INIT"], valstep=1)
speed_slider.label.set_color("white")
speed_slider.valtext.set_color("white")

ax = fig.add_axes([0.35, 0.06, 0.63, 0.88], projection="polar", facecolor="black")
ax.set_ylim(0, env.r_max)
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.spines["polar"].set_edgecolor("white")
ax.spines["polar"].set_linewidth(1.6)
ax.set_title("Hack vs Dove (essential + energetic satiety)", color="white", pad=18)

ring_th = np.linspace(0, TWOPI, 800)
ax.plot(ring_th, np.full_like(ring_th, env.r_max), color="white", linewidth=2.0, alpha=0.95)

food_sc = ax.scatter([], [], s=28, c="green", alpha=0.65)
cell_sc = ax.scatter([], [], s=20, c="dodgerblue", alpha=0.85)


def format_death_causes(dc: dict):
    total = sum(dc.values()) if dc else 0
    if total == 0:
        return "death causes: (no deaths yet)"
    order = sorted(dc.items(), key=lambda kv: kv[1], reverse=True)
    lines = ["death causes (share):"]
    for k, v in order[:5]:
        lines.append(f"  {k:12s}: {v:5d} ({100.0*v/total:5.1f}%)")
    return "\n".join(lines)


def animate(_):
    if sim.just_started_phase:
        sim.just_started_phase = False
    else:
        steps = int(speed_slider.val)
        for _ in range(steps):
            sim.step()
            if sim.step() is not None:
                break
            if pop.size() == 0:
                break

    # food
    if int(env.food_x.size) > 0:
        fr, fth = env.cart_to_polar(env.food_x, env.food_y)
        food_sc.set_offsets(np.c_[to_cpu(fth), to_cpu(fr)])
        food_sc.set_sizes(np.full(int(env.food_x.size), 28.0))
    else:
        food_sc.set_offsets(np.empty((0, 2)))

    # cells
    if pop.size() == 0:
        cell_sc.set_offsets(np.empty((0, 2)))
        stats_txt.set_text("Extinción total.\n")
        return food_sc, cell_sc, stats_txt

    r, th = env.cart_to_polar(pop.x, pop.y)
    e = to_cpu(pop.energy)
    t = to_cpu(pop.type)
    done = to_cpu(pop.done_for_day)
    ready = to_cpu(pop.ready_to_repro)
    ate = to_cpu(pop.ate_today)

    colors = np.array(["red" if x == 1 else "dodgerblue" for x in t])
    sizes = np.clip(e, 10, 180)

    cell_sc.set_offsets(np.c_[to_cpu(th), to_cpu(r)])
    cell_sc.set_sizes(sizes)
    cell_sc.set_color(colors)

    total = pop.size()
    hawks = int((t == 1).sum())
    doves = total - hawks
    ph = 100.0 * hawks / total
    pd = 100.0 * doves / total

    pct_done = 100.0 * float(done.mean())
    pct_ready = 100.0 * float(ready.mean())
    pct_ate = 100.0 * float(ate.mean())
    pct_sated = 100.0 * float((e >= (P["SATIETY_E"] - P["SATIETY_EPS"])).mean())

    stats_txt.set_text(
        f"phase: {sim.phase}  step: {sim.phase_step}/{P['DAY_STEPS'] if sim.phase=='day' else P['NIGHT_STEPS']}\n"
        f"day: {sim.day}\n"
        f"pop: {total:4d}   births_today: {sim.stats['births_today']:4d}   deaths_today: {sim.stats['deaths_today']:4d}\n"
        f"Hawk: {hawks:4d} ({ph:5.1f}%)   Dove: {doves:4d} ({pd:5.1f}%)\n"
        f"E(mean/med): {e.mean():7.2f} / {np.median(e):7.2f}\n"
        f"food active: {int(env.food_x.size):4d}   eaten_today: {sim.stats['eaten_today']:4d}\n"
        f"\n"
        f"per-cell % (today)\n"
        f"  ate >=1      : {pct_ate:6.1f}%\n"
        f"  sated (E>=S) : {pct_sated:6.1f}%  (S={P['SATIETY_E']})\n"
        f"  done_for_day : {pct_done:6.1f}%\n"
        f"  ready_repro  : {pct_ready:6.1f}%  (goal={P['REPRO_GOAL']})\n"
        f"\n"
        f"V={P['FOOD_VALUE_V']}  D={P['INJURY_COST_D']}  T={P['TIME_COST_T']}\n"
        f"move_cost={P['MOVE_COST_PER_DIST']}/dist  metab(day/night)={P['METABOLIC_DAY']}/{P['METABOLIC_NIGHT']}\n"
        f"repro_thr={P['REPRO_THRESHOLD']}  food/day={P['FOOD_PER_DAY']}  r_min={P['FOOD_SPAWN_R_MIN']}\n"
        f"\n"
        f"{format_death_causes(sim.stats['death_causes'])}\n"
    )

    return food_sc, cell_sc, stats_txt


ani = FuncAnimation(fig, animate, frames=2500, interval=45, blit=False)
plt.show()

