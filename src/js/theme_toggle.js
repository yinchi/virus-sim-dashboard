(switchOn) => {
    // Set dark or light mode based on switchOn (boolean, dark = true)
    document.documentElement.setAttribute(
        'data-mantine-color-scheme', switchOn ? 'dark' : 'light');  
    return window.dash_clientside.no_update
}
