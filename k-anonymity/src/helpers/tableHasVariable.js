export default function tableHasVariable(variable, table, variablesByTableName){
    return (variablesByTableName[table.name] || []).some(({ name }) => name === variable.name);
}