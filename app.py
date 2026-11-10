import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

st.set_page_config(
    page_title="Flower Federated Learning Simulation",
    page_icon="🌼",
    layout="wide"
)

st.title("🌼 Flower Federated Learning Dashboard")
st.caption("Simulasi konsep Federated Learning dengan 1 Server dan 3 Client menggunakan agregasi FedAvg.")

with st.sidebar:
    st.header("Pengaturan Simulasi")
    rounds = st.slider("Jumlah ronde pelatihan", 1, 20, 5)
    start_button = st.button("▶️ Mulai Simulasi", type="primary", use_container_width=True)
    reset_button = st.button("🔄 Reset", use_container_width=True)
    st.divider()
    st.caption("Demo edukasi: data tetap berada pada masing-masing client; server hanya menerima bobot model simulasi.")

if "history" not in st.session_state:
    st.session_state.history = None
if reset_button:
    st.session_state.history = None
    st.rerun()

def run_simulation(n_rounds):
    rng = np.random.default_rng(42)
    client_sizes = [120, 100, 80]
    client_names = ["Client 1", "Client 2", "Client 3"]
    global_accuracy = 0.55
    global_loss = 1.25
    rows = []
    client_rows = []

    for r in range(1, n_rounds + 1):
        local_accs = []
        local_losses = []
        for i, name in enumerate(client_names):
            improvement = 0.025 + (i * 0.003) + rng.uniform(-0.006, 0.008)
            local_acc = min(0.98, global_accuracy + improvement)
            local_loss = max(0.08, global_loss - 0.09 - i * 0.008 + rng.uniform(-0.025, 0.02))
            local_accs.append(local_acc)
            local_losses.append(local_loss)
            client_rows.append({
                "Ronde": r,
                "Client": name,
                "Jumlah data lokal": client_sizes[i],
                "Akurasi lokal": round(local_acc * 100, 2),
                "Loss lokal": round(local_loss, 4),
                "Status": "Completed"
            })

        weights = np.array(client_sizes, dtype=float)
        weights /= weights.sum()
        global_accuracy = float(np.dot(local_accs, weights))
        global_loss = float(np.dot(local_losses, weights))
        rows.append({
            "Ronde": r,
            "Akurasi global": round(global_accuracy * 100, 2),
            "Loss global": round(global_loss, 4)
        })

    return pd.DataFrame(rows), pd.DataFrame(client_rows)

if start_button:
    progress = st.progress(0, text="Menyiapkan server dan client...")
    status = st.empty()
    for pct, message in [(20, "Server menginisialisasi model global..."),
                         (45, "Tiga client melakukan pelatihan lokal..."),
                         (70, "Client mengirim bobot model ke server..."),
                         (90, "Server menjalankan agregasi FedAvg..."),
                         (100, "Simulasi selesai.")]:
        progress.progress(pct, text=message)
        status.info(message)
    st.session_state.history = run_simulation(rounds)
    status.success("Simulasi berhasil diselesaikan.")
    progress.empty()

st.subheader("Topologi Federated Learning")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Server", "1", "Coordinator")
with c2:
    st.metric("Client aktif", "3", "Client 1, 2, 3")
with c3:
    st.metric("Strategi agregasi", "FedAvg", "Federated Averaging")

st.markdown("""
<div style="display:flex;justify-content:center;align-items:center;gap:12px;flex-wrap:wrap;margin:14px 0 20px 0;">
  <div style="background:#17365D;color:white;padding:18px 30px;border-radius:12px;text-align:center;min-width:180px;">
    <div style="font-size:26px;">🖥️</div><b>SERVER</b><br><small>Global Model & FedAvg</small>
  </div>
  <div style="font-size:28px;color:#64748b;">↕</div>
  <div style="display:flex;gap:10px;flex-wrap:wrap;justify-content:center;">
    <div style="background:#e0f2fe;color:#075985;padding:15px 20px;border-radius:12px;text-align:center;"><div style="font-size:24px;">💻</div><b>CLIENT 1</b><br><small>Data lokal 120</small></div>
    <div style="background:#dcfce7;color:#166534;padding:15px 20px;border-radius:12px;text-align:center;"><div style="font-size:24px;">💻</div><b>CLIENT 2</b><br><small>Data lokal 100</small></div>
    <div style="background:#fef3c7;color:#92400e;padding:15px 20px;border-radius:12px;text-align:center;"><div style="font-size:24px;">💻</div><b>CLIENT 3</b><br><small>Data lokal 80</small></div>
  </div>
</div>
""", unsafe_allow_html=True)

if st.session_state.history is None:
    st.info("Atur jumlah ronde pada sidebar, lalu klik **Mulai Simulasi** untuk melihat proses pelatihan dan agregasi.")
else:
    history, client_history = st.session_state.history
    last = history.iloc[-1]
    st.subheader("Hasil Simulasi")
    m1, m2, m3 = st.columns(3)
    m1.metric("Ronde selesai", f"{len(history)}")
    m2.metric("Akurasi global akhir", f"{last['Akurasi global']:.2f}%")
    m3.metric("Loss global akhir", f"{last['Loss global']:.4f}")

    left, right = st.columns(2)
    with left:
        fig_acc = px.line(history, x="Ronde", y="Akurasi global", markers=True,
                          title="Perkembangan Akurasi Global (%)")
        fig_acc.update_layout(yaxis_title="Akurasi (%)", xaxis_title="Ronde")
        st.plotly_chart(fig_acc, use_container_width=True)
    with right:
        fig_loss = px.line(history, x="Ronde", y="Loss global", markers=True,
                           title="Perkembangan Loss Global")
        fig_loss.update_layout(yaxis_title="Loss", xaxis_title="Ronde")
        st.plotly_chart(fig_loss, use_container_width=True)

    st.subheader("Aktivitas Client")
    st.dataframe(client_history, use_container_width=True, hide_index=True)
    st.subheader("Riwayat Model Global")
    st.dataframe(history, use_container_width=True, hide_index=True)

    csv = client_history.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Unduh hasil simulasi (CSV)", csv,
                       "hasil_simulasi_flower.csv", "text/csv")

st.divider()
st.subheader("Alur Kerja Simulasi")
st.markdown("""
1. **Server** menginisialisasi model global dan mengirim parameter awal.
2. **Client 1, Client 2, dan Client 3** melatih model menggunakan data lokal masing-masing.
3. Setiap client mengirimkan pembaruan bobot model ke server, bukan data mentah.
4. **Server** menggabungkan bobot berdasarkan jumlah data lokal menggunakan konsep *Federated Averaging (FedAvg)*.
5. Model global diperbarui dan proses berulang sampai ronde terakhir.
""")
st.warning("Catatan akademik: aplikasi ini merupakan simulasi visual dan numerik konsep Flower/FedAvg untuk demonstrasi. Aplikasi ini belum menjalankan proses federated learning terdistribusi melalui Flower runtime atau Flower Dashboard.")
st.caption("Proyek praktikum — Federated Learning | Streamlit Community Cloud")
