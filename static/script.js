<script>
    document.getElementById("toggleButton").addEventListener("click", function() {
        var sidebar = document.getElementById("sidebar");
        var mainContent = document.getElementById("mainContent");
        sidebar.classList.toggle("collapsed");
        mainContent.classList.toggle("collapsed");
    });
</script>
