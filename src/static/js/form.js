document.addEventListener("DOMContentLoaded", () => {
    const classeSelect = document.getElementById("classe");
    const annoSelect = document.getElementById("anno");

    function updateAnnoOptions() {
        let maxAnno = 5;
        if (classeSelect.value === "sec1") maxAnno = 3;
        if (classeSelect.value === "sec2") maxAnno = 5;

        annoSelect.innerHTML = "";
        for (let i = 1; i <= maxAnno; i++) {
            const option = document.createElement("option");
            option.value = i;
            option.textContent = i;
            annoSelect.appendChild(option);
        }
    }

    classeSelect.addEventListener("change", updateAnnoOptions);
    updateAnnoOptions();
});
